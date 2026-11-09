import { useCallback, useEffect, useId, useRef, useState, type FormEvent } from 'react'
import type { SerieTemporal } from './SelectorSerie'

type Seleccion = { conjuntoId: number; serie: SerieTemporal }
type Props = { seleccion: Seleccion | null; activo: boolean }
const TECNICAS = [
  { id: 'estacional_semanal_recursiva', nombre: 'Línea base estacional semanal' },
  { id: 'random_forest', nombre: 'Random Forest' },
  { id: 'hist_gradient_boosting', nombre: 'HistGradientBoosting' },
] as const
const CORTES = ['2017-06-16', '2017-07-01', '2017-07-16'] as const
type Tecnica = typeof TECNICAS[number]['id']
type Corte = typeof CORTES[number]

type GrupoVariables = 'clasico' | 'ampliado'
const VARIABLES_CLASICAS = ['lag_1', 'lag_7', 'lag_14', 'media_7', 'dia_semana', 'mes'] as const
const VARIABLES_AMPLIADAS = [...VARIABLES_CLASICAS, 'media_14'] as const
type ParametrosSolicitados = Record<string, number | null>
type BorradorRF = { n_estimators: string; max_depth: string; min_samples_leaf: string }
type BorradorHGB = { max_iter: string; learning_rate: string; max_depth: string }
// Valores y límites de configuracion_experimentos.py; el backend sigue validando.
const INICIAL_RF: BorradorRF = { n_estimators: '100', max_depth: '', min_samples_leaf: '5' }
const INICIAL_HGB: BorradorHGB = { max_iter: '100', learning_rate: '0.05', max_depth: '' }

function numeroEnRango(texto: string, minimo: number, maximo: number, entero = true): number | null {
  const valor = Number(texto)
  return texto.trim() !== '' && Number.isFinite(valor) && valor >= minimo && valor <= maximo &&
    (!entero || Number.isInteger(valor)) ? valor : null
}

function resolverParametros(tecnica: Tecnica, rf: BorradorRF, hgb: BorradorHGB): ParametrosSolicitados | null {
  if (tecnica === 'estacional_semanal_recursiva') return {}
  const textoProfundidad = tecnica === 'random_forest' ? rf.max_depth : hgb.max_depth
  const profundidad = textoProfundidad.trim() === '' ? null : numeroEnRango(textoProfundidad, 1, 50)
  if (textoProfundidad.trim() !== '' && profundidad === null) return null
  let valores: ParametrosSolicitados
  let predeterminados: ParametrosSolicitados
  if (tecnica === 'random_forest') {
    const arboles = numeroEnRango(rf.n_estimators, 1, 500)
    const hoja = numeroEnRango(rf.min_samples_leaf, 1, 100)
    if (arboles === null || hoja === null) return null
    valores = { n_estimators: arboles, max_depth: profundidad, min_samples_leaf: hoja }
    predeterminados = { n_estimators: Number(INICIAL_RF.n_estimators), max_depth: null,
      min_samples_leaf: Number(INICIAL_RF.min_samples_leaf) }
  } else {
    const iteraciones = numeroEnRango(hgb.max_iter, 1, 500)
    const tasa = numeroEnRango(hgb.learning_rate, 0.001, 1, false)
    if (iteraciones === null || tasa === null) return null
    valores = { max_iter: iteraciones, learning_rate: tasa, max_depth: profundidad }
    predeterminados = { max_iter: Number(INICIAL_HGB.max_iter), learning_rate: Number(INICIAL_HGB.learning_rate), max_depth: null }
  }
  // Solo diferencias efectivas: 100.0 equivale a 100; vacío en profundidad es null.
  return Object.fromEntries(Object.entries(valores).filter(([nombre, valor]) => valor !== predeterminados[nombre]))
}

type SolicitudEnCurso = {
  conjuntoId: number; serie: SerieTemporal; tecnica: Tecnica; corte: Corte
  grupo: GrupoVariables | null; parametros: ParametrosSolicitados
}
type Metadatos = {
  id: number; experimento_id: number; tecnica: Tecnica; fecha_corte: Corte
  tipo: 'validacion'; horizonte: number; fecha_generacion: string
  inicio_entrenamiento: string; fin_entrenamiento: string
}
type Referencia = Pick<Metadatos, 'id' | 'experimento_id' | 'tecnica' | 'fecha_corte'>
type Fila = { paso: number; fecha_objetivo: string; venta_estimada: number }
type RespuestaDetalle = { conjunto_id: number; serie: SerieTemporal; pronostico: Metadatos; detalles: Fila[] }
type EstadoLista = { kind: 'loading' } | { kind: 'error'; message: string } | { kind: 'success'; filas: Metadatos[] }
type EstadoDetalle = { kind: 'idle' } | { kind: 'loading'; id: number } |
  { kind: 'error'; message: string } | { kind: 'success'; data: RespuestaDetalle }
type EstadoGeneracion = { kind: 'idle' } | { kind: 'success' | 'error'; message: string }
const DIA = 86_400_000
// Formato de presentación; no se redondean ni reemplazan valores en el estado.
const formatoVentas = new Intl.NumberFormat('es-EC', { maximumFractionDigits: 3 })
const formatoCientifico = new Intl.NumberFormat('es-EC', { notation: 'scientific', maximumFractionDigits: 3 })
const formatoGeneracion = new Intl.DateTimeFormat('es-EC', { dateStyle: 'medium', timeStyle: 'short' })

function presentarVenta(valor: number): string {
  // Un positivo que se redondearía a cero conserva su orden de magnitud visible.
  return valor !== 0 && (valor < 0.001 || valor >= 1_000_000_000)
    ? formatoCientifico.format(valor) : formatoVentas.format(valor)
}

function registro(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
function idValido(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0
}
function esTecnica(value: unknown): value is Tecnica {
  return TECNICAS.some(t => t.id === value)
}
function esCorte(value: unknown): value is Corte {
  return CORTES.some(c => c === value)
}
function instanteFecha(value: unknown): number | null {
  if (typeof value !== 'string' || !/^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(value)) return null
  const instante = Date.parse(`${value}T00:00:00Z`)
  return Number.isFinite(instante) && new Date(instante).toISOString().slice(0, 10) === value ? instante : null
}
function fechaFutura(corte: Corte, paso: number): string {
  return new Date((instanteFecha(corte) as number) + paso * DIA).toISOString().slice(0, 10)
}
function mismaSeleccion(value: unknown, seleccion: Seleccion): boolean {
  if (!registro(value) || !registro(value.serie)) return false
  return value.conjunto_id === seleccion.conjuntoId && value.serie.id === seleccion.serie.id &&
    value.serie.store_nbr === seleccion.serie.store_nbr && value.serie.family === seleccion.serie.family
}
function esMetadato(value: unknown): value is Metadatos {
  if (!registro(value)) return false
  return idValido(value.id) && idValido(value.experimento_id) && esTecnica(value.tecnica) &&
    esCorte(value.fecha_corte) && value.tipo === 'validacion' && value.horizonte === 15 &&
    typeof value.fecha_generacion === 'string' && Number.isFinite(Date.parse(value.fecha_generacion)) &&
    instanteFecha(value.inicio_entrenamiento) !== null && instanteFecha(value.fin_entrenamiento) !== null &&
    (value.inicio_entrenamiento as string) <= (value.fin_entrenamiento as string) &&
    (value.fin_entrenamiento as string) <= value.fecha_corte
}
function esDetalle(value: unknown, seleccion: Seleccion, esperado: Referencia): value is RespuestaDetalle {
  if (!registro(value) || !mismaSeleccion(value, seleccion) || !esMetadato(value.pronostico) ||
      !Array.isArray(value.detalles) || value.detalles.length !== 15) return false
  const p = value.pronostico
  if (p.id !== esperado.id || p.experimento_id !== esperado.experimento_id ||
      p.tecnica !== esperado.tecnica || p.fecha_corte !== esperado.fecha_corte) return false
  return value.detalles.every((fila: unknown, indice: number) => registro(fila) &&
    fila.paso === indice + 1 && fila.fecha_objetivo === fechaFutura(p.fecha_corte, indice + 1) &&
    typeof fila.venta_estimada === 'number' && Number.isFinite(fila.venta_estimada) && fila.venta_estimada >= 0)
}
function nombreTecnica(tecnica: Tecnica): string {
  return TECNICAS.find(t => t.id === tecnica)?.nombre ?? tecnica
}

function GraficoPronostico({ detalles, serie, pronostico }: {
  detalles: Fila[]; serie: SerieTemporal; pronostico: Metadatos
}) {
  const id = useId()
  const ancho = 760
  const alto = 340
  const izquierda = 122
  const derecha = 48
  const arriba = 24
  const abajo = 66
  const anchoGrafico = ancho - izquierda - derecha
  const altoGrafico = alto - arriba - abajo
  // Este componente solo recibe los quince detalles que ya superaron esDetalle.
  const inicio = instanteFecha(detalles[0].fecha_objetivo) as number
  const fin = instanteFecha(detalles[14].fecha_objetivo) as number
  const maximoReal = detalles.reduce((maximo, fila) => Math.max(maximo, fila.venta_estimada), 0)
  const maximo = maximoReal > 0 ? maximoReal : 1
  const posicionX = (fecha: string) => izquierda +
    ((instanteFecha(fecha) as number) - inicio) / (fin - inicio) * anchoGrafico
  // Dividir antes de multiplicar evita desbordamientos con valores grandes.
  // Con máximo positivo tampoco se resta un mínimo igual (casos constantes).
  const posicionY = (valor: number) => arriba + (1 - valor / maximo) * altoGrafico
  const marcasY = [...new Set([0, 0.25, 0.5, 0.75, 1].map(fraccion => maximo * fraccion))]
  const marcasX = [0, 3, 7, 10, 14].map(indice => detalles[indice])
  const puntos = detalles.map(fila => `${posicionX(fila.fecha_objetivo)},${posicionY(fila.venta_estimada)}`).join(' ')

  return <figure className="forecast-chart">
    <figcaption className="forecast-chart-caption">
      <strong>Ventas diarias estimadas</strong>
      <span>Tienda {serie.store_nbr} · {serie.family} · {nombreTecnica(pronostico.tecnica)}</span>
      <span>Corte {pronostico.fecha_corte} · Periodo {detalles[0].fecha_objetivo} a {detalles[14].fecha_objetivo}</span>
    </figcaption>
    <div className="forecast-chart-scroll" tabIndex={0} role="region" aria-label="Gráfico de las quince ventas estimadas, desplazable horizontalmente">
      <svg viewBox={`0 0 ${ancho} ${alto}`} role="img" aria-labelledby={`${id}-title ${id}-desc`}>
        <title id={`${id}-title`}>Quince ventas diarias estimadas de la tienda {serie.store_nbr}, {serie.family}</title>
        <desc id={`${id}-desc`}>
          Eje horizontal: fecha objetivo. Eje vertical lineal desde cero: ventas estimadas.
          La línea conecta los quince valores recibidos. Los ceros usan puntos huecos y los positivos rellenos.
          La tabla inferior ofrece los mismos datos y permite consultar cada valor completo.
        </desc>
        {marcasY.map(valor => <g key={valor}>
          <line x1={izquierda} x2={ancho - derecha} y1={posicionY(valor)} y2={posicionY(valor)} className="forecast-chart-grid" />
          <text x={izquierda - 10} y={posicionY(valor) + 4} textAnchor="end" className="forecast-chart-label">{presentarVenta(valor)}</text>
        </g>)}
        <line x1={izquierda} x2={izquierda} y1={arriba} y2={alto - abajo} className="forecast-chart-axis" />
        <line x1={izquierda} x2={ancho - derecha} y1={alto - abajo} y2={alto - abajo} className="forecast-chart-axis" />
        {marcasX.map(fila => <text key={fila.paso} x={posicionX(fila.fecha_objetivo)} y={alto - abajo + 23}
          textAnchor="middle" className="forecast-chart-label">{fila.fecha_objetivo}</text>)}
        <text x={izquierda + anchoGrafico / 2} y={alto - 7} textAnchor="middle" className="forecast-chart-axis-title">Fecha objetivo</text>
        <text transform={`translate(18 ${arriba + altoGrafico / 2}) rotate(-90)`} textAnchor="middle" className="forecast-chart-axis-title">Ventas estimadas</text>
        <polyline points={puntos} className="forecast-chart-line" />
        {detalles.map(fila => <circle key={fila.paso} cx={posicionX(fila.fecha_objetivo)} cy={posicionY(fila.venta_estimada)} r="4"
          className={`forecast-chart-point${fila.venta_estimada === 0 ? ' forecast-chart-point--zero' : ''}`}>
          <title>{fila.fecha_objetivo}: valor completo {String(fila.venta_estimada)}</title>
        </circle>)}
      </svg>
    </div>
    <p className="forecast-chart-note">Puntos huecos: estimación igual a cero. Puntos rellenos: valor positivo. Estimaciones basadas en ventas históricas.</p>
  </figure>
}

class ErrorSolicitud extends Error {}
async function leerRespuesta(response: Response): Promise<unknown> {
  let body: unknown
  try { body = await response.json() } catch {
    throw new ErrorSolicitud(`El servidor respondió con HTTP ${response.status}, pero no devolvió un JSON válido.`)
  }
  if (!response.ok) {
    const mensaje = registro(body) && registro(body.detail) && typeof body.detail.codigo === 'string' &&
      typeof body.detail.mensaje === 'string' && body.detail.mensaje.trim() ? body.detail.mensaje : null
    throw new ErrorSolicitud(mensaje ? `${mensaje} (HTTP ${response.status}).` : `La solicitud respondió con HTTP ${response.status}.`)
  }
  return body
}
function mensajeError(error: unknown, alternativa: string): string {
  return error instanceof ErrorSolicitud ? error.message : alternativa
}

type PanelProps = {
  seleccion: Seleccion; activo: boolean; pendiente: boolean; revision: number
  iniciar: (solicitud: SolicitudEnCurso) => boolean
  finalizar: (cambioSeleccion: boolean) => void
}

function PanelPronosticos({ seleccion, activo, pendiente, revision, iniciar, finalizar }: PanelProps) {
  const { conjuntoId, serie: { id: serieId, store_nbr: tienda, family: familia } } = seleccion
  const [tecnica, setTecnica] = useState<Tecnica>('estacional_semanal_recursiva')
  const [corte, setCorte] = useState<Corte>('2017-06-16')
  const [grupo, setGrupo] = useState<GrupoVariables>('clasico')
  const [rf, setRf] = useState<BorradorRF>(() => ({ ...INICIAL_RF }))
  const [hgb, setHgb] = useState<BorradorHGB>(() => ({ ...INICIAL_HGB }))
  const esLineaBase = tecnica === 'estacional_semanal_recursiva'
  const parametros = resolverParametros(tecnica, rf, hgb)
  const modificados = parametros !== null && Object.keys(parametros).length > 0
  const [lista, setLista] = useState<EstadoLista>({ kind: 'loading' })
  const [detalle, setDetalle] = useState<EstadoDetalle>({ kind: 'idle' })
  const [generacion, setGeneracion] = useState<EstadoGeneracion>({ kind: 'idle' })
  const solicitudLista = useRef<AbortController | null>(null)
  const solicitudDetalle = useRef<AbortController | null>(null)
  const vivo = useRef(true)
  const ultimaRevision = useRef(-1)

  useEffect(() => {
    vivo.current = true
    return () => {
      vivo.current = false
      ultimaRevision.current = -1 // permite repetir el GET abortado por StrictMode
      solicitudLista.current?.abort()
      solicitudDetalle.current?.abort()
    }
  }, [])

  const consultarLista = useCallback(async () => {
    solicitudLista.current?.abort()
    const controller = new AbortController()
    solicitudLista.current = controller
    setLista({ kind: 'loading' })
    try {
      const response = await fetch(`/api/pronosticos/active/series/${serieId}`, { signal: controller.signal })
      const body = await leerRespuesta(response)
      if (!vivo.current || controller.signal.aborted) return
      const contexto = { conjuntoId, serie: { id: serieId, store_nbr: tienda, family: familia } }
      if (!registro(body) || !mismaSeleccion(body, contexto) || !Array.isArray(body.pronosticos) ||
          !body.pronosticos.every(esMetadato) ||
          new Set(body.pronosticos.map(p => p.id)).size !== body.pronosticos.length) {
        throw new ErrorSolicitud('El listado recibido no corresponde a la serie y conjunto seleccionados o contiene datos inválidos.')
      }
      setLista({ kind: 'success', filas: body.pronosticos })
    } catch (error) {
      if (vivo.current && !controller.signal.aborted) setLista({ kind: 'error', message:
        mensajeError(error, 'No se pudo conectar para consultar los pronósticos almacenados.') })
    } finally {
      if (solicitudLista.current === controller) solicitudLista.current = null
    }
  }, [conjuntoId, serieId, tienda, familia])

  useEffect(() => {
    if (activo && ultimaRevision.current !== revision) {
      ultimaRevision.current = revision
      void consultarLista()
    }
  }, [activo, revision, consultarLista])

  async function consultarDetalle(esperado: Referencia) {
    solicitudDetalle.current?.abort()
    const controller = new AbortController()
    solicitudDetalle.current = controller
    setDetalle({ kind: 'loading', id: esperado.id })
    try {
      const response = await fetch(`/api/pronosticos/active/${esperado.id}`, { signal: controller.signal })
      const body = await leerRespuesta(response)
      if (!vivo.current || controller.signal.aborted) return
      if (!esDetalle(body, seleccion, esperado)) throw new ErrorSolicitud(
        'El detalle recibido no corresponde al pronóstico solicitado o no contiene quince pasos, fechas y estimaciones válidos.')
      setDetalle({ kind: 'success', data: body })
    } catch (error) {
      if (vivo.current && !controller.signal.aborted) setDetalle({ kind: 'error', message:
        mensajeError(error, 'No se pudo conectar para consultar el detalle del pronóstico.') })
    } finally {
      if (solicitudDetalle.current === controller) solicitudDetalle.current = null
    }
  }

  async function generar(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!esTecnica(tecnica) || !esCorte(corte) || parametros === null || !event.currentTarget.reportValidity() ||
        !iniciar({ conjuntoId, serie: seleccion.serie, tecnica, corte, grupo: esLineaBase ? null : grupo, parametros })) return
    const configuracion = !esLineaBase && (grupo === 'ampliado' || modificados) ? {
      tecnica, version_contrato: grupo === 'ampliado' ? 2 : 1, version_reglas: grupo === 'ampliado' ? 2 : 1,
      variables: grupo === 'ampliado' ? VARIABLES_AMPLIADAS : VARIABLES_CLASICAS,
      ...(modificados ? { parametros } : {}),
    } : null
    setGeneracion({ kind: 'idle' })
    try {
      // Solo la acción de formulario inicia un POST. No se aborta ni reintenta:
      // cerrar una vista no implica cancelar el trabajo que recibe el servidor.
      const response = await fetch(`/api/pronosticos/active/series/${serieId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tecnica, fecha_corte: corte, ...(configuracion ? { configuracion } : {}) }),
      })
      const body = await leerRespuesta(response)
      if (!vivo.current) return
      if (!registro(body) || ![200, 201].includes(response.status) || body.conjunto_id !== conjuntoId ||
          body.serie_id !== serieId || body.tecnica !== tecnica || body.fecha_corte !== corte ||
          body.horizonte !== 15 || body.cantidad_detalles !== 15 || !idValido(body.pronostico_id) ||
          !idValido(body.experimento_id) || body.nuevo !== (response.status === 201) ||
          body.reutilizado !== (response.status === 200)) throw new ErrorSolicitud(
        'No se pudo confirmar una respuesta consistente de generación. Consulte el listado antes de volver a generar.')
      setGeneracion({ kind: 'success', message: response.status === 201
        ? 'Se creó un pronóstico nuevo.' : 'Se reutilizó un pronóstico existente.' })
      await Promise.all([consultarLista(), consultarDetalle({ id: body.pronostico_id,
        experimento_id: body.experimento_id, tecnica, fecha_corte: corte })])
    } catch (error) {
      if (vivo.current) setGeneracion({ kind: 'error', message: mensajeError(error,
        'No se pudo confirmar la generación por un error de red. El servidor puede seguir procesándola. Consulte el listado antes de volver a generar.') })
    } finally {
      finalizar(!vivo.current)
    }
  }

  return (
    <>
      <p className="forecast-series">Tienda <strong>{tienda}</strong> · Familia <strong>{familia}</strong></p>
      <form className="forecast-form" onSubmit={event => { void generar(event) }} aria-busy={pendiente}>
        <div>
          <label htmlFor="forecast-technique">Técnica</label>
          <select id="forecast-technique" value={tecnica} disabled={pendiente}
            onChange={event => {
              if (!pendiente && esTecnica(event.target.value)) {
                setTecnica(event.target.value)
                if (event.target.value === 'estacional_semanal_recursiva') setGrupo('clasico')
              }
            }}>
            {TECNICAS.map(t => <option key={t.id} value={t.id}>{t.nombre}</option>)}
          </select>
        </div>
        <div>
          <label htmlFor="forecast-cutoff">Fecha de corte</label>
          <select id="forecast-cutoff" value={corte} disabled={pendiente}
            onChange={event => { if (esCorte(event.target.value)) setCorte(event.target.value) }}>
            {CORTES.map((c, i) => <option key={c} value={c}>{c} · Validación {i + 1}</option>)}
          </select>
        </div>
        <div className="forecast-variable-group">
          <label htmlFor="forecast-variable-group">Grupo de variables predictoras</label>
          <select id="forecast-variable-group" value={grupo} disabled={pendiente || esLineaBase}
            aria-describedby="forecast-variable-help"
            onChange={event => {
              if (!pendiente && !esLineaBase && (event.target.value === 'clasico' || event.target.value === 'ampliado')) {
                setGrupo(event.target.value)
              }
            }}>
            {esLineaBase ? <option value="clasico">No aplica a la línea base</option> : <>
              <option value="clasico">Clásico — 6 variables</option>
              <option value="ampliado">Ampliado — 7 variables</option>
            </>}
          </select>
        </div>
        <div id="forecast-variable-help" className="help-text forecast-variable-help">
          {esLineaBase ? <p>La línea base semanal no utiliza un grupo de variables predictoras configurable.</p> : <>
            <p>Variables del grupo seleccionado: <code>{(grupo === 'ampliado' ? VARIABLES_AMPLIADAS : VARIABLES_CLASICAS).join(', ')}</code>.</p>
            <p>El grupo ampliado añade <code>media_14</code>: emplea las catorce fechas naturales anteriores y requiere
              observaciones para esas fechas. Las fechas sin registro no se sustituyen por ventas cero.</p>
          </>}
        </div>
        {!esLineaBase && <fieldset className="forecast-parameters" disabled={pendiente}
          aria-describedby="forecast-parameters-help">
          <legend>Hiperparámetros de {nombreTecnica(tecnica)}</legend>
          <div className="forecast-parameter-fields">
            {tecnica === 'random_forest' ? <>
              <div>
                <label htmlFor="forecast-n-estimators">Número de árboles (n_estimators)</label>
                <input id="forecast-n-estimators" type="number" min="1" max="500" step="1" required
                  value={rf.n_estimators} aria-describedby="forecast-n-estimators-help"
                  aria-invalid={numeroEnRango(rf.n_estimators, 1, 500) === null}
                  onChange={event => { if (!pendiente) setRf({ ...rf, n_estimators: event.target.value }) }} />
                <p id="forecast-n-estimators-help" className="help-text">Entero de 1 a 500. Predeterminado: 100.</p>
              </div>
              <div>
                <label htmlFor="forecast-min-samples-leaf">Mínimo de muestras por hoja (min_samples_leaf)</label>
                <input id="forecast-min-samples-leaf" type="number" min="1" max="100" step="1" required
                  value={rf.min_samples_leaf} aria-describedby="forecast-min-samples-leaf-help"
                  aria-invalid={numeroEnRango(rf.min_samples_leaf, 1, 100) === null}
                  onChange={event => { if (!pendiente) setRf({ ...rf, min_samples_leaf: event.target.value }) }} />
                <p id="forecast-min-samples-leaf-help" className="help-text">Entero de 1 a 100. Predeterminado: 5.</p>
              </div>
            </> : <>
              <div>
                <label htmlFor="forecast-max-iter">Iteraciones (max_iter)</label>
                <input id="forecast-max-iter" type="number" min="1" max="500" step="1" required
                  value={hgb.max_iter} aria-describedby="forecast-max-iter-help"
                  aria-invalid={numeroEnRango(hgb.max_iter, 1, 500) === null}
                  onChange={event => { if (!pendiente) setHgb({ ...hgb, max_iter: event.target.value }) }} />
                <p id="forecast-max-iter-help" className="help-text">Entero de 1 a 500. Predeterminado: 100.</p>
              </div>
              <div>
                <label htmlFor="forecast-learning-rate">Tasa de aprendizaje (learning_rate)</label>
                <input id="forecast-learning-rate" type="number" min="0.001" max="1" step="any" required
                  value={hgb.learning_rate} aria-describedby="forecast-learning-rate-help"
                  aria-invalid={numeroEnRango(hgb.learning_rate, 0.001, 1, false) === null}
                  onChange={event => { if (!pendiente) setHgb({ ...hgb, learning_rate: event.target.value }) }} />
                <p id="forecast-learning-rate-help" className="help-text">Entre 0,001 y 1. Predeterminado: 0,05.</p>
              </div>
            </>}
            <div>
              <label htmlFor="forecast-max-depth">Profundidad máxima (max_depth)</label>
              <input id="forecast-max-depth" type="number" min="1" max="50" step="1" placeholder="Sin límite"
                value={tecnica === 'random_forest' ? rf.max_depth : hgb.max_depth}
                aria-describedby="forecast-max-depth-help"
                aria-invalid={(tecnica === 'random_forest' ? rf.max_depth : hgb.max_depth).trim() !== '' &&
                  numeroEnRango(tecnica === 'random_forest' ? rf.max_depth : hgb.max_depth, 1, 50) === null}
                onChange={event => {
                  if (!pendiente) {
                    if (tecnica === 'random_forest') setRf({ ...rf, max_depth: event.target.value })
                    else setHgb({ ...hgb, max_depth: event.target.value })
                  }
                }} />
              <p id="forecast-max-depth-help" className="help-text">Vacío: sin límite, el valor predeterminado. O entero de 1 a 50.</p>
            </div>
          </div>
          <p id="forecast-parameters-help" className="help-text">Los demás parámetros conservan sus valores predeterminados. Los valores se validan antes de generar.</p>
          <p className="forecast-parameter-state" role="status">{parametros === null
            ? 'Hay parámetros inválidos. Revisa los límites indicados antes de generar.'
            : modificados ? 'Configuración de hiperparámetros modificada.' : 'Hiperparámetros predeterminados.'}</p>
          <button type="button" onClick={() => {
            if (!pendiente) {
              if (tecnica === 'random_forest') setRf({ ...INICIAL_RF })
              else setHgb({ ...INICIAL_HGB })
            }
          }}>Restablecer parámetros de {nombreTecnica(tecnica)}</button>
        </fieldset>}
        <p className="help-text">Horizonte fijo: quince días naturales desde el día siguiente al corte.</p>
        <button type="submit" disabled={pendiente || parametros === null || !esTecnica(tecnica) || !esCorte(corte)}>Generar pronóstico</button>
      </form>
      {generacion.kind !== 'idle' && <p className={`forecast-message forecast-message--${generacion.kind}`}
        role={generacion.kind === 'error' ? 'alert' : 'status'}>{generacion.message}</p>}

      <section className="forecast-results" aria-labelledby="forecast-list-heading">
        <div className="forecast-list-header">
          <h3 id="forecast-list-heading">Pronósticos almacenados</h3>
          <button type="button" onClick={() => { void consultarLista() }} disabled={lista.kind === 'loading'}>Actualizar listado</button>
        </div>
        {lista.kind === 'loading' && <p role="status">Cargando pronósticos almacenados…</p>}
        {lista.kind === 'error' && <p className="forecast-message forecast-message--error" role="alert">{lista.message}</p>}
        {lista.kind === 'success' && (lista.filas.length === 0
          ? <p role="status">No hay pronósticos almacenados para esta serie.</p>
          : <div className="forecast-table-wrap" tabIndex={0} role="region" aria-label="Listado de pronósticos">
            <table>
              <caption>Resultados disponibles de la serie seleccionada</caption>
              <thead><tr><th scope="col">Técnica</th><th scope="col">Corte</th><th scope="col">Periodo (15 días)</th>
                <th scope="col">Generación</th><th scope="col">Acción</th></tr></thead>
              <tbody>{lista.filas.map(p => <tr key={p.id}>
                <td>{nombreTecnica(p.tecnica)}</td><td>{p.fecha_corte}</td>
                <td>{fechaFutura(p.fecha_corte, 1)} a {fechaFutura(p.fecha_corte, 15)}</td>
                <td>{formatoGeneracion.format(new Date(p.fecha_generacion))}</td>
                <td><button type="button" disabled={detalle.kind === 'loading' && detalle.id === p.id}
                  aria-label={`Ver pronóstico de ${nombreTecnica(p.tecnica)} con corte ${p.fecha_corte}`}
                  onClick={() => { void consultarDetalle(p) }}>Ver pronóstico</button></td>
              </tr>)}</tbody>
            </table>
          </div>)}
      </section>
      {detalle.kind === 'loading' && <p className="forecast-message" role="status">Consultando detalle del pronóstico…</p>}
      {detalle.kind === 'error' && <p className="forecast-message forecast-message--error" role="alert">{detalle.message}</p>}
      {detalle.kind === 'success' && <section className="forecast-results" aria-labelledby="forecast-detail-heading">
        <h3 id="forecast-detail-heading">{nombreTecnica(detalle.data.pronostico.tecnica)}</h3>
        <p>Tienda {tienda} · {familia} · Corte {detalle.data.pronostico.fecha_corte}</p>
        <p>Periodo: {detalle.data.detalles[0].fecha_objetivo} a {detalle.data.detalles[14].fecha_objetivo}.</p>
        <p className="forecast-explanation">Estimaciones basadas en ventas históricas.</p>
        <GraficoPronostico detalles={detalle.data.detalles} serie={detalle.data.serie} pronostico={detalle.data.pronostico} />
        <p className="forecast-value-help">Se muestran hasta tres decimales o notación científica para magnitudes extremas. Abra «Valor completo» para consultar el número recibido.</p>
        <div className="forecast-table-wrap" tabIndex={0} role="region" aria-label="Quince estimaciones diarias">
          <table>
            <caption>Pronóstico diario de demanda</caption>
            <thead><tr><th scope="col">Paso</th><th scope="col">Fecha objetivo</th><th scope="col">Venta estimada</th></tr></thead>
            <tbody>{detalle.data.detalles.map(fila => <tr key={fila.paso}>
              <td>{fila.paso}</td><td>{fila.fecha_objetivo}</td><td>
                <span>{presentarVenta(fila.venta_estimada)}</span>
                {fila.venta_estimada !== 0 && <details className="forecast-value-details">
                  <summary aria-label={`Valor completo del paso ${fila.paso}`}>Valor completo</summary>
                  <span>{String(fila.venta_estimada)}</span>
                </details>}
              </td>
            </tr>)}</tbody>
          </table>
        </div>
      </section>}
    </>
  )
}

function PronosticosVentas({ seleccion, activo }: Props) {
  // Este contenedor no se reinicia al cambiar de selección: conserva el bloqueo
  // del POST. Solo el panel con resultados usa una clave de conjunto/serie.
  const [generando, setGenerando] = useState<SolicitudEnCurso | null>(null)
  const [revision, setRevision] = useState(0)
  const bloqueo = useRef(false)
  const iniciar = useCallback((solicitud: SolicitudEnCurso) => {
    if (bloqueo.current) return false
    bloqueo.current = true
    setGenerando(solicitud)
    return true
  }, [])
  const finalizar = useCallback((cambioSeleccion: boolean) => {
    bloqueo.current = false
    setGenerando(null)
    if (cambioSeleccion) setRevision(v => v + 1)
  }, [])

  return <section className="forecast-card" aria-labelledby="forecast-heading">
    <h2 id="forecast-heading">Pronósticos de demanda</h2>
    {generando !== null && <div className="forecast-message" role="status">
      <p>Generando pronóstico. El entrenamiento puede tardar algunos segundos.</p>
      <p>Solicitud para conjunto {generando.conjuntoId} · tienda {generando.serie.store_nbr} · {generando.serie.family}.</p>
      <p>{nombreTecnica(generando.tecnica)} · Corte {generando.corte} · {generando.grupo === null
        ? 'Sin grupo configurable' : generando.grupo === 'ampliado' ? 'Ampliado — 7 variables' : 'Clásico — 6 variables'}.</p>
      {generando.grupo !== null && <p>{Object.keys(generando.parametros).length === 0 ? 'Hiperparámetros predeterminados.'
        : `Parámetros solicitados: ${Object.entries(generando.parametros).map(([nombre, valor]) => `${nombre}=${valor}`).join(', ')}.`}</p>}
    </div>}
    {seleccion === null
      ? <p role="status">Seleccione una tienda y una familia válidas para consultar o generar pronósticos.</p>
      : <PanelPronosticos key={`${seleccion.conjuntoId}:${seleccion.serie.id}`} seleccion={seleccion}
        activo={activo} pendiente={generando !== null} revision={revision} iniciar={iniciar} finalizar={finalizar} />}
  </section>
}

export default PronosticosVentas
