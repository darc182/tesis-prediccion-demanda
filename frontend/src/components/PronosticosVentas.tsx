import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
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
const formatoVentas = new Intl.NumberFormat('es-EC', { maximumSignificantDigits: 17 })
const formatoGeneracion = new Intl.DateTimeFormat('es-EC', { dateStyle: 'medium', timeStyle: 'short' })

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
  iniciar: (serie: SerieTemporal) => boolean
  finalizar: (cambioSeleccion: boolean) => void
}

function PanelPronosticos({ seleccion, activo, pendiente, revision, iniciar, finalizar }: PanelProps) {
  const { conjuntoId, serie: { id: serieId, store_nbr: tienda, family: familia } } = seleccion
  const [tecnica, setTecnica] = useState<Tecnica>('estacional_semanal_recursiva')
  const [corte, setCorte] = useState<Corte>('2017-06-16')
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
    if (!esTecnica(tecnica) || !esCorte(corte) || !iniciar(seleccion.serie)) return
    setGeneracion({ kind: 'idle' })
    try {
      // Solo la acción de formulario inicia un POST. No se aborta ni reintenta:
      // cerrar una vista no implica cancelar el trabajo que recibe el servidor.
      const response = await fetch(`/api/pronosticos/active/series/${serieId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tecnica, fecha_corte: corte }),
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
            onChange={event => { if (esTecnica(event.target.value)) setTecnica(event.target.value) }}>
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
        <p className="help-text">Horizonte fijo: quince días naturales desde el día siguiente al corte.</p>
        <button type="submit" disabled={pendiente || !esTecnica(tecnica) || !esCorte(corte)}>Generar pronóstico</button>
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
        <div className="forecast-table-wrap" tabIndex={0} role="region" aria-label="Quince estimaciones diarias">
          <table>
            <caption>Pronóstico diario de demanda</caption>
            <thead><tr><th scope="col">Paso</th><th scope="col">Fecha objetivo</th><th scope="col">Venta estimada</th></tr></thead>
            <tbody>{detalle.data.detalles.map(fila => <tr key={fila.paso}>
              <td>{fila.paso}</td><td>{fila.fecha_objetivo}</td><td>{formatoVentas.format(fila.venta_estimada)}</td>
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
  const [generando, setGenerando] = useState<SerieTemporal | null>(null)
  const [revision, setRevision] = useState(0)
  const bloqueo = useRef(false)
  const iniciar = useCallback((serie: SerieTemporal) => {
    if (bloqueo.current) return false
    bloqueo.current = true
    setGenerando(serie)
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
      <p>Solicitud para tienda {generando.store_nbr} · {generando.family}.</p>
    </div>}
    {seleccion === null
      ? <p role="status">Seleccione una tienda y una familia válidas para consultar o generar pronósticos.</p>
      : <PanelPronosticos key={`${seleccion.conjuntoId}:${seleccion.serie.id}`} seleccion={seleccion}
        activo={activo} pendiente={generando !== null} revision={revision} iniciar={iniciar} finalizar={finalizar} />}
  </section>
}

export default PronosticosVentas
