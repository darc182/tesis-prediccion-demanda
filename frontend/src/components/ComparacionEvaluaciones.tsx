import { useEffect, useId, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import type { SerieTemporal, SeriesState } from './SelectorSerie'
import ConsolidacionEvaluaciones from './ConsolidacionEvaluaciones'
import type { DestinoSimulacion } from './Simulaciones'
import ConfiguracionPronosticoGuardada from './ConfiguracionPronosticoGuardada'
import { generacionListadoValida, identificarCandidato, useConfiguracionGuardada,
  type CandidatoConfiguracion } from './consultaConfiguracionPronostico'

type Seleccion = { conjuntoId: number; serie: SerieTemporal }
type Props = {
  seleccion: Seleccion | null; activo: boolean; estadoConjunto: SeriesState['kind']
  simular: (destino: DestinoSimulacion) => void
}
const CORTES = ['2017-06-16', '2017-07-01', '2017-07-16'] as const
type Corte = typeof CORTES[number]
const TECNICAS = {
  estacional_semanal_recursiva: 'Línea base estacional semanal',
  random_forest: 'Random Forest',
  hist_gradient_boosting: 'HistGradientBoosting',
} as const
type Tecnica = keyof typeof TECNICAS
type Opcion = CandidatoConfiguracion & { fecha_corte: Corte }
type Eleccion = { corte: Corte; opciones: Opcion[] }
type Detalle = {
  paso: number; fecha_objetivo: string; venta_estimada: number
  venta_real: number | null; error_absoluto: number | null
}
type Evaluacion = {
  fecha_corte: Corte; horizonte: 15; observaciones_evaluadas: number; fechas_sin_registro: number
  cobertura: number; definicion_cobertura: string; fechas_ausentes: string[]
  mae: number | null; wape: number | null; rmse: number | null; detalles: Detalle[]
}
type Resultado = { pronostico_id: number; experimento_id: number; tecnica: Tecnica; evaluacion: Evaluacion }
type Comparacion = {
  conjunto_id: number; serie: SerieTemporal; fecha_corte: Corte; horizonte: 15
  estado_cobertura: 'completa' | 'parcial' | 'sin_observaciones'; alcance_metricas: string
  observaciones_evaluadas: number; cobertura: number; fechas_ausentes: string[]; resultados: Resultado[]
}
type Lista = { kind: 'loading' } | { kind: 'error'; mensaje: string } |
  { kind: 'success'; opciones: Opcion[]; total: number }
type Consulta = { kind: 'idle' | 'loading' } | { kind: 'error'; mensaje: string } |
  { kind: 'success'; datos: Comparacion }

const ERRORES: Readonly<Record<string, string>> = {
  entrada_invalida: 'Seleccione dos o tres pronósticos distintos del mismo corte.',
  pronostico_invalido: 'Uno de los identificadores seleccionados no es válido.',
  pronostico_inexistente: 'Uno de los pronósticos ya no existe. Actualice el listado.',
  pronostico_inactivo: 'Uno de los pronósticos no pertenece al conjunto activo. Actualice la selección.',
  sin_conjunto_activo: 'No hay un conjunto de datos activo para esta consulta.',
  pronosticos_no_comparables: 'Los pronósticos no comparten serie, periodo y observaciones reales. Revise la selección.',
  datos_inconsistentes: 'Los datos almacenados están incompletos o son inconsistentes.',
  tipo_no_evaluable: 'Solo se admiten pronósticos de validación.',
  corte_no_autorizado: 'La fecha de corte no pertenece a las tres ventanas de validación.',
  base_no_disponible: 'La base de datos no está disponible. Intente una nueva consulta más tarde.',
  error_interno: 'No se pudo completar la consulta.',
  serie_inexistente: 'La serie seleccionada ya no existe en el conjunto activo.',
  serie_inactiva: 'La serie no pertenece al conjunto activo. Actualice la selección.',
  pronostico_corrupto: 'El listado contiene un pronóstico incompleto o inconsistente.',
}
const RESPUESTA_INVALIDA = 'La respuesta recibida está incompleta o no corresponde a la selección. No se muestran resultados.'
const ALCANCES = {
  completa: 'Métricas sobre las quince observaciones del horizonte.',
  parcial: 'Métricas sobre observaciones disponibles.',
  sin_observaciones: 'Sin observaciones reales; métricas no definidas.',
} as const
const formato = new Intl.NumberFormat('es-EC', { maximumSignificantDigits: 6 })
const formatoEje = new Intl.NumberFormat('es-EC', { maximumSignificantDigits: 3, notation: 'compact' })
const registro = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)
const idValido = (v: unknown): v is number => typeof v === 'number' && Number.isSafeInteger(v) && v > 0
const numero = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= 0
const tecnica = (v: unknown): v is Tecnica => typeof v === 'string' && Object.hasOwn(TECNICAS, v)
const corteValido = (v: unknown): v is Corte => CORTES.some(c => c === v)
const mismasFechas = (a: unknown, b: string[]): a is string[] => Array.isArray(a) && a.length === b.length && a.every((v, i) => v === b[i])
function claves(v: Record<string, unknown>, nombres: string): boolean {
  const esperadas = nombres.split(' ')
  return Object.keys(v).length === esperadas.length && esperadas.every(k => Object.hasOwn(v, k))
}
function fechaValida(v: unknown): v is string {
  if (typeof v !== 'string' || !/^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(v)) return false
  const t = Date.parse(`${v}T00:00:00Z`)
  return Number.isFinite(t) && new Date(t).toISOString().slice(0, 10) === v
}
function fechaObjetivo(corte: Corte, paso: number): string {
  return new Date(Date.parse(`${corte}T00:00:00Z`) + paso * 86_400_000).toISOString().slice(0, 10)
}
function mismoContexto(v: Record<string, unknown>, seleccion: Seleccion): boolean {
  return v.conjunto_id === seleccion.conjuntoId && registro(v.serie) &&
    claves(v.serie, 'id store_nbr family') && v.serie.id === seleccion.serie.id &&
    v.serie.store_nbr === seleccion.serie.store_nbr && v.serie.family === seleccion.serie.family
}
function mismaOpcion(a: Opcion, b: Opcion): boolean {
  return a.id === b.id && a.experimento_id === b.experimento_id && a.tecnica === b.tecnica && a.fecha_corte === b.fecha_corte
}

// El listado puede contener prueba/futuro y otros horizontes: se validan sus
// metadatos, pero solo se ofrecen las tres ventanas de validación de quince días.
function leerLista(v: unknown, seleccion: Seleccion): { opciones: Opcion[]; total: number } | null {
  if (!registro(v) || !mismoContexto(v, seleccion) || !Array.isArray(v.pronosticos)) return null
  const ids = new Set<number>()
  const opciones: Opcion[] = []
  for (const p of v.pronosticos) {
    if (!registro(p) || !idValido(p.id) || !idValido(p.experimento_id) || !tecnica(p.tecnica) ||
        !fechaValida(p.fecha_corte) || !['validacion', 'prueba', 'futuro'].includes(p.tipo as string) ||
        !idValido(p.horizonte) || ids.has(p.id)) return null
    ids.add(p.id)
    if (p.tipo === 'validacion' && p.horizonte === 15 && corteValido(p.fecha_corte)) {
      if (!generacionListadoValida(p.fecha_generacion)) return null
      opciones.push({ id: p.id, experimento_id: p.experimento_id, tecnica: p.tecnica, fecha_corte: p.fecha_corte,
        horizonte: p.horizonte, fecha_generacion: p.fecha_generacion })
    }
  }
  // Orden de presentación estable; el orden de selección se conserva aparte.
  return { opciones: opciones.sort((a, b) => a.id - b.id), total: v.pronosticos.length }
}

function esEvaluacion(v: unknown, corte: Corte): v is Evaluacion {
  if (!registro(v) || !claves(v, 'fecha_corte horizonte observaciones_evaluadas fechas_sin_registro cobertura definicion_cobertura mae wape rmse detalles fechas_ausentes') ||
      v.fecha_corte !== corte || v.horizonte !== 15 ||
      v.definicion_cobertura !== 'observaciones_evaluadas / horizonte (15 fechas naturales)' ||
      !Array.isArray(v.detalles) || v.detalles.length !== 15) return false
  const ausentes: string[] = []
  let hayRealesPositivos = false
  for (const [i, d] of v.detalles.entries()) {
    if (!registro(d) || !claves(d, 'paso fecha_objetivo venta_estimada venta_real error_absoluto') ||
        d.paso !== i + 1 || d.fecha_objetivo !== fechaObjetivo(corte, i + 1) || !numero(d.venta_estimada)) return false
    if (d.venta_real === null) {
      if (d.error_absoluto !== null) return false
      ausentes.push(d.fecha_objetivo as string)
    } else {
      if (!numero(d.venta_real) || !numero(d.error_absoluto)) return false
      hayRealesPositivos ||= d.venta_real > 0
    }
  }
  const cantidad = 15 - ausentes.length
  if (v.observaciones_evaluadas !== cantidad || v.fechas_sin_registro !== ausentes.length ||
      v.cobertura !== cantidad / 15 || !mismasFechas(v.fechas_ausentes, ausentes)) return false
  // Solo coherencia del contrato: no se recalculan errores ni MAE/WAPE/RMSE.
  if (cantidad === 0) return v.mae === null && v.wape === null && v.rmse === null
  return numero(v.mae) && numero(v.rmse) && (hayRealesPositivos ? numero(v.wape) : v.wape === null)
}
function esComparacion(v: unknown, seleccion: Seleccion, eleccion: Eleccion): v is Comparacion {
  if (!registro(v) || !claves(v, 'conjunto_id serie fecha_corte horizonte estado_cobertura alcance_metricas observaciones_evaluadas cobertura fechas_ausentes resultados') ||
      !mismoContexto(v, seleccion) || v.fecha_corte !== eleccion.corte || v.horizonte !== 15 ||
      !Array.isArray(v.resultados) || v.resultados.length !== eleccion.opciones.length ||
      v.resultados.length < 2 || v.resultados.length > 3) return false
  let referencia: Evaluacion | null = null
  for (const [i, r] of v.resultados.entries()) {
    const p = eleccion.opciones[i]
    if (!registro(r) || !claves(r, 'pronostico_id experimento_id tecnica evaluacion') ||
        r.pronostico_id !== p.id || r.experimento_id !== p.experimento_id || r.tecnica !== p.tecnica ||
        !esEvaluacion(r.evaluacion, eleccion.corte)) return false
    const e = r.evaluacion
    if (referencia && e.detalles.some((d, j) => d.venta_real !== referencia!.detalles[j].venta_real)) return false
    referencia = e
    if (v.observaciones_evaluadas !== e.observaciones_evaluadas || v.cobertura !== e.cobertura ||
        !mismasFechas(v.fechas_ausentes, e.fechas_ausentes)) return false
  }
  const estado = referencia!.observaciones_evaluadas === 15 ? 'completa' :
    referencia!.observaciones_evaluadas === 0 ? 'sin_observaciones' : 'parcial'
  return v.estado_cobertura === estado && v.alcance_metricas === ALCANCES[estado]
}
function mensajeHttp(v: unknown): string {
  if (registro(v) && registro(v.detail) && typeof v.detail.codigo === 'string' && Object.hasOwn(ERRORES, v.detail.codigo)) {
    return ERRORES[v.detail.codigo]
  }
  return 'No se pudo completar la consulta. Revise la conexión y vuelva a solicitarla cuando lo desee.'
}
async function obtener(url: string, signal: AbortSignal): Promise<unknown> {
  let respuesta: Response
  try { respuesta = await fetch(url, { method: 'GET', signal, cache: 'no-store' }) }
  catch { throw new Error('No se pudo conectar para consultar los datos. Compruebe la conexión con FastAPI.') }
  let body: unknown
  try { body = await respuesta.json() }
  catch { throw new Error('La API no devolvió un JSON válido. No se muestran resultados.') }
  if (!respuesta.ok) throw new Error(mensajeHttp(body))
  return body
}
function etiqueta(p: Opcion | Resultado): string {
  return `${TECNICAS[p.tecnica]} · pronóstico ${'id' in p ? p.id : p.pronostico_id} · experimento ${p.experimento_id}`
}
function Valor({ valor, ausente = 'No definido' }: { valor: number | null; ausente?: string }) {
  if (valor === null) return <span>{ausente}</span>
  const visible = formato.format(valor)
  if (visible === String(valor)) return <span>{visible}</span>
  return <><span title={String(valor)}>{visible}</span><details className="evaluation-value">
    <summary>Valor completo</summary><code>{String(valor)}</code>
  </details></>
}

function Grafico({ datos, id }: { datos: Comparacion; id: string }) {
  const dias = datos.resultados[0].evaluacion.detalles
  const maximo = Math.max(...datos.resultados.flatMap(r => r.evaluacion.detalles.map(d => d.venta_estimada)),
    ...dias.map(d => d.venta_real ?? 0)) || 1
  const x = (i: number) => 80 + i / 14 * 870
  // Dividir antes de multiplicar conserva una escala finita con valores grandes.
  const y = (v: number) => 270 - v / maximo * 230
  const reales = dias.map((d, i) => {
    if (d.venta_real === null) return ''
    return `${i > 0 && dias[i - 1].venta_real !== null ? 'L' : 'M'}${x(i)},${y(d.venta_real)}`
  }).join(' ')
  return <figure className="evaluation-chart">
    <figcaption>Ventas reales y estimadas · {dias[0].fecha_objetivo} a {dias[14].fecha_objetivo}</figcaption>
    <div className="evaluation-scroll" tabIndex={0} role="region" aria-label="Gráfico desplazable de quince fechas">
      <svg viewBox="0 0 1000 365" role="img" aria-labelledby={`${id}-titulo ${id}-desc`}>
        <title id={`${id}-titulo`}>Ventas reales frente a los pronósticos seleccionados</title>
        <desc id={`${id}-desc`}>Quince fechas con ejes compartidos. Reales con círculos; estimaciones con cuadrados,
          trazos diferenciados y leyenda. Las fechas sin registro interrumpen la curva real.
          Los valores completos están en la tabla diaria siguiente.</desc>
        {[0, 1, 2, 3, 4].map(i => <g key={i}>
          <line className="evaluation-grid" x1="80" x2="950" y1={y(maximo * (i / 4))} y2={y(maximo * (i / 4))} />
          <text x="70" y={y(maximo * (i / 4)) + 5} textAnchor="end">{formatoEje.format(maximo * (i / 4))}</text>
        </g>)}
        <text x="80" y="22">Ventas</text>
        {dias.map((d, i) => <text key={d.paso} transform={`translate(${x(i)},290) rotate(-45)`} textAnchor="end">
          {d.fecha_objetivo.slice(5)}</text>)}
        <text x="520" y="352" textAnchor="middle">Fecha objetivo (mes-día)</text>
        {datos.resultados.map((r, indice) => <g key={r.pronostico_id} className={`evaluation-curve evaluation-curve--${indice}`}>
          <polyline points={r.evaluacion.detalles.map((d, i) => `${x(i)},${y(d.venta_estimada)}`).join(' ')} />
          {r.evaluacion.detalles.map((d, i) => <rect key={d.paso} x={x(i) - 3} y={y(d.venta_estimada) - 3} width="6" height="6">
            <title>{etiqueta(r)} · {d.fecha_objetivo}: {String(d.venta_estimada)}</title>
          </rect>)}
        </g>)}
        <g className="evaluation-curve evaluation-curve--real"><path d={reales} />
          {dias.map((d, i) => d.venta_real === null ? null : <circle key={d.paso} cx={x(i)} cy={y(d.venta_real)} r="4">
            <title>Venta real · {d.fecha_objetivo}: {String(d.venta_real)}</title>
          </circle>)}
        </g>
      </svg>
    </div>
    <ul className="evaluation-legend" aria-label="Leyenda del gráfico">
      <li><svg viewBox="0 0 50 14" aria-hidden="true" className="evaluation-curve evaluation-curve--real"><path d="M0,7 H50" /><circle cx="25" cy="7" r="4" /></svg>Ventas reales</li>
      {datos.resultados.map((r, i) => <li key={r.pronostico_id}><svg viewBox="0 0 50 14" aria-hidden="true" className={`evaluation-curve evaluation-curve--${i}`}>
        <path d="M0,7 H50" /><rect x="22" y="4" width="6" height="6" /></svg>{etiqueta(r)}</li>)}
    </ul>
    <p>Los huecos son fechas sin registro, no ventas cero. La tabla diaria ofrece la alternativa al gráfico.</p>
  </figure>
}

function Resultados({ datos, simular }: { datos: Comparacion; simular: Props['simular'] }) {
  const id = useId()
  const primera = datos.resultados[0].evaluacion
  return <section aria-label="Resultado de la comparación" className="evaluation-results">
    <h3>Evaluaciones del corte {datos.fecha_corte}</h3>
    <p>Conjunto {datos.conjunto_id} · tienda {datos.serie.store_nbr} · {datos.serie.family} · horizonte de 15 días.</p>
    <div className="evaluation-notice" role="status">
      <strong>Cobertura {datos.estado_cobertura === 'sin_observaciones' ? 'sin observaciones' : datos.estado_cobertura}:
        {' '}{datos.observaciones_evaluadas}/15 observaciones.</strong>
      <p>{ALCANCES[datos.estado_cobertura]}</p>
      {datos.estado_cobertura === 'sin_observaciones' && <p>Las estimaciones están disponibles, pero no hay ventas reales para determinar el desempeño predictivo.</p>}
      {datos.fechas_ausentes.length > 0 && <p>{datos.fechas_ausentes.length} fechas sin registro: {datos.fechas_ausentes.join(', ')}.</p>}
    </div>
    <div className="evaluation-scroll" tabIndex={0} role="region" aria-label="Resumen de métricas desplazable">
      <table><caption>Métricas sobre las mismas fechas observadas</caption>
        <thead><tr><th scope="col">Técnica e identificación</th><th scope="col">MAE</th><th scope="col">WAPE (%)</th>
          <th scope="col">RMSE</th><th scope="col">Observaciones evaluadas</th><th scope="col">Cobertura</th>
          <th scope="col">Simulación</th></tr></thead>
        <tbody>{datos.resultados.map(r => <tr key={r.pronostico_id}><th scope="row">{etiqueta(r)}</th>
          <td><Valor valor={r.evaluacion.mae} /></td><td><Valor valor={r.evaluacion.wape} /></td><td><Valor valor={r.evaluacion.rmse} /></td>
          <td>{r.evaluacion.observaciones_evaluadas}/15</td><td><Valor valor={r.evaluacion.cobertura} /> (proporción)</td>
          <td><button type="button" aria-label={`Simular con este pronóstico: pronóstico ${r.pronostico_id}`}
            onClick={() => simular({
              conjuntoId: datos.conjunto_id, serie: datos.serie, pronosticoId: r.pronostico_id,
              experimentoId: r.experimento_id, tecnica: r.tecnica, corte: datos.fecha_corte,
            })}>Simular con este pronóstico</button></td></tr>)}</tbody>
      </table>
    </div>
    <p>MAE es el error absoluto medio. WAPE es el error absoluto ponderado, expresado en porcentaje;
      puede superar 100 y no está definido si las ventas observadas suman cero. RMSE es la raíz del error
      cuadrático medio y da mayor peso a errores grandes. Cero es un valor válido; «No definido» representa null.</p>
    <p>La cobertura es observaciones evaluadas / 15. Con cobertura parcial, las métricas describen únicamente
      las mismas fechas disponibles. Esta pantalla no guarda evaluaciones ni selecciona un ganador.</p>
    <Grafico datos={datos} id={id} />
    <div className="evaluation-scroll" tabIndex={0} role="region" aria-label="Detalle diario desplazable">
      <table><caption>Detalle diario de las quince fechas · alternativa tabular del gráfico</caption>
        <thead><tr><th scope="col" rowSpan={2}>Paso</th><th scope="col" rowSpan={2}>Fecha objetivo</th><th scope="col" rowSpan={2}>Venta real</th>
          {datos.resultados.map(r => <th scope="colgroup" colSpan={2} key={r.pronostico_id}>{etiqueta(r)}</th>)}</tr>
          <tr>{datos.resultados.flatMap(r => [<th scope="col" key={`${r.pronostico_id}-venta`} aria-label={`Venta estimada · ${etiqueta(r)}`}>Venta estimada</th>,
            <th scope="col" key={`${r.pronostico_id}-error`} aria-label={`Error absoluto · ${etiqueta(r)}`}>Error absoluto</th>])}</tr></thead>
        <tbody>{primera.detalles.map((d, i) => <tr key={d.paso}><th scope="row">{d.paso}</th><td><time dateTime={d.fecha_objetivo}>{d.fecha_objetivo}</time></td>
          <td><Valor valor={d.venta_real} ausente="Sin registro" /></td>
          {datos.resultados.flatMap(r => [<td key={`${r.pronostico_id}-venta`}><Valor valor={r.evaluacion.detalles[i].venta_estimada} /></td>,
            <td key={`${r.pronostico_id}-error`}><Valor valor={r.evaluacion.detalles[i].error_absoluto} ausente="No evaluable" /></td>])}</tr>)}</tbody>
      </table>
    </div>
  </section>
}

type PanelProps = {
  seleccion: Seleccion; eleccion: Eleccion; setEleccion: Dispatch<SetStateAction<Eleccion>>
  simular: Props['simular']
}
function PanelComparacion({ seleccion, eleccion, setEleccion, simular }: PanelProps) {
  const [lista, setLista] = useState<Lista>({ kind: 'loading' })
  const [consulta, setConsulta] = useState<Consulta>({ kind: 'idle' })
  const [revision, setRevision] = useState(0)
  const inspector = useConfiguracionGuardada(seleccion, lista.kind === 'success'
    ? JSON.stringify([seleccion.conjuntoId, seleccion.serie.id, seleccion.serie.store_nbr, seleccion.serie.family, eleccion.corte, revision]) : null)
  const listado = useRef<AbortController | null>(null)
  const comparacion = useRef<AbortController | null>(null)
  const identidad = useRef(0)
  const { conjuntoId, serie: { id: serieId, store_nbr: tienda, family: familia } } = seleccion
  const id = useId()

  useEffect(() => {
    const controller = new AbortController()
    listado.current = controller
    async function cargar() {
      setLista({ kind: 'loading' })
      try {
        const body = await obtener(`/api/pronosticos/active/series/${serieId}`, controller.signal)
        if (controller.signal.aborted || listado.current !== controller) return
        const datos = leerLista(body, { conjuntoId, serie: { id: serieId, store_nbr: tienda, family: familia } })
        if (!datos) throw new Error(RESPUESTA_INVALIDA)
        setEleccion(previa => ({ ...previa, opciones: previa.opciones.filter(p => datos.opciones.some(n => mismaOpcion(p, n))) }))
        setLista({ kind: 'success', ...datos })
      } catch (error) {
        if (!controller.signal.aborted && listado.current === controller) {
          setLista({ kind: 'error', mensaje: error instanceof Error ? error.message : RESPUESTA_INVALIDA })
        }
      }
    }
    void cargar()
    return () => {
      controller.abort()
      comparacion.current?.abort()
    }
  }, [conjuntoId, serieId, tienda, familia, revision, setEleccion])

  function invalidar() {
    comparacion.current?.abort()
    comparacion.current = null
    identidad.current++
    setConsulta({ kind: 'idle' })
  }
  function cambiarCorte(corte: Corte) {
    invalidar()
    setEleccion({ corte, opciones: [] })
  }
  function alternar(p: Opcion) {
    invalidar()
    setEleccion(anterior => ({ ...anterior, opciones: anterior.opciones.some(o => o.id === p.id)
      ? anterior.opciones.filter(o => o.id !== p.id) : anterior.opciones.length < 3 ? [...anterior.opciones, p] : anterior.opciones }))
  }
  const disponibles = lista.kind === 'success' ? lista.opciones.filter(p => p.fecha_corte === eleccion.corte) : []
  const seleccionValida = eleccion.opciones.length >= 2 && eleccion.opciones.length <= 3 &&
    new Set(eleccion.opciones.map(p => p.id)).size === eleccion.opciones.length &&
    eleccion.opciones.every(p => disponibles.some(d => mismaOpcion(p, d)))

  async function comparar() {
    if (!seleccionValida || comparacion.current) return
    const controller = new AbortController()
    comparacion.current = controller
    const solicitud = ++identidad.current
    const pedida = { corte: eleccion.corte, opciones: [...eleccion.opciones] }
    setConsulta({ kind: 'loading' })
    const parametros = new URLSearchParams()
    pedida.opciones.forEach(p => parametros.append('pronostico_id', String(p.id)))
    try {
      const body = await obtener(`/api/evaluaciones/comparacion/active?${parametros}`, controller.signal)
      if (controller.signal.aborted || comparacion.current !== controller || identidad.current !== solicitud) return
      if (!esComparacion(body, seleccion, pedida)) throw new Error(RESPUESTA_INVALIDA)
      setConsulta({ kind: 'success', datos: body })
    } catch (error) {
      if (!controller.signal.aborted && comparacion.current === controller && identidad.current === solicitud) {
        setConsulta({ kind: 'error', mensaje: error instanceof Error ? error.message : RESPUESTA_INVALIDA })
      }
    } finally {
      if (comparacion.current === controller) comparacion.current = null
    }
  }
  return <>
    <p>Consulte métricas de dos o tres pronósticos existentes para una misma serie y un mismo corte.</p>
    <div className="evaluation-actions"><button type="button" onClick={() => { invalidar(); setLista({ kind: 'loading' }); setRevision(r => r + 1) }} disabled={lista.kind === 'loading'}>Actualizar listado</button></div>
    {lista.kind === 'loading' && <p role="status">Cargando pronósticos almacenados…</p>}
    {lista.kind === 'error' && <p className="evaluation-notice" role="alert">{lista.mensaje}</p>}
    {lista.kind === 'success' && <>
      {lista.total === 0 ? <p>No hay pronósticos almacenados para esta serie.</p> : lista.opciones.length === 0 ?
        <p>No hay pronósticos elegibles. Solo se comparan validaciones de quince días en los tres cortes autorizados.</p> : null}
      <label className="evaluation-cut" htmlFor={`${id}-corte`}>Fecha de corte de validación
        <select id={`${id}-corte`} value={eleccion.corte} onChange={e => { if (corteValido(e.target.value)) cambiarCorte(e.target.value) }}>
          {CORTES.map(c => <option key={c} value={c}>{c} · {lista.opciones.filter(p => p.fecha_corte === c).length} pronósticos</option>)}
        </select></label>
      {disponibles.length === 0 && <p>No hay pronósticos elegibles para este corte.</p>}
      {disponibles.length === 1 && <p>Solo hay un pronóstico elegible en este corte. Se necesitan al menos dos.</p>}
      {lista.opciones.length >= 2 && !CORTES.some(c => lista.opciones.filter(p => p.fecha_corte === c).length >= 2) &&
        <p>No existe una combinación comparable: los pronósticos disponibles pertenecen a cortes distintos.</p>}
      <fieldset className="evaluation-options" aria-describedby={`${id}-ayuda`}><legend>Pronósticos disponibles para {eleccion.corte}</legend>
        {disponibles.map(p => {
          const posicion = eleccion.opciones.findIndex(o => o.id === p.id)
          return <div className="evaluation-candidate" key={p.id}>
            <label><input type="checkbox" checked={posicion >= 0}
              disabled={posicion < 0 && eleccion.opciones.length === 3} onChange={() => alternar(p)} />
              <span>{identificarCandidato(p)}{posicion >= 0 && <strong> · posición {posicion + 1}</strong>}</span>
            </label>
            <button type="button" aria-label={`Ver configuración del pronóstico ${p.id}`}
              disabled={inspector.consulta?.estado.kind === 'loading' && inspector.consulta.pronostico.id === p.id}
              onClick={() => { void inspector.consultar(p, String(p.id)) }}>Ver configuración</button>
            <ConfiguracionPronosticoGuardada inspector={inspector} ubicacion={String(p.id)} />
          </div>
        })}
      </fieldset>
      <p id={`${id}-ayuda`}>Elija dos o tres. El orden de selección será el orden de los resultados; desmarque para cambiarlo.</p>
      <ol className="evaluation-order" aria-label="Orden de selección">{eleccion.opciones.map(p => <li key={p.id}>{etiqueta(p)}</li>)}</ol>
      <div className="evaluation-actions"><button type="button" disabled={!seleccionValida || consulta.kind === 'loading'} onClick={() => { void comparar() }}>Comparar evaluaciones</button></div>
    </>}
    {consulta.kind === 'loading' && <p role="status">Consultando métricas y ventas observadas…</p>}
    {consulta.kind === 'error' && <p className="evaluation-notice" role="alert">{consulta.mensaje}</p>}
    {lista.kind === 'success' && consulta.kind === 'success' && <Resultados datos={consulta.datos} simular={simular} />}
  </>
}

export default function ComparacionEvaluaciones({ seleccion, activo, estadoConjunto, simular }: Props) {
  // App asigna una key por conjunto y serie: un cambio de contexto descarta
  // también esta elección. Al salir solo se desmonta el panel de solicitudes.
  const [eleccion, setEleccion] = useState<Eleccion>({ corte: CORTES[0], opciones: [] })
  const [modalidad, setModalidad] = useState<'corte' | 'consolidada'>('corte')
  const idModalidad = useId()
  if (!activo) return null
  return <section className="evaluation-card" aria-label="Comparación cuantitativa">
    <h2>Comparación de evaluaciones</h2>
    <div className="consolidation-modes" role="group" aria-label="Modalidad de comparación">
      <button type="button" aria-pressed={modalidad === 'corte'} aria-controls={`${idModalidad}-panel`}
        onClick={() => setModalidad('corte')}>Por corte — 15 días</button>
      <button type="button" aria-pressed={modalidad === 'consolidada'} aria-controls={`${idModalidad}-panel`}
        onClick={() => setModalidad('consolidada')}>Consolidada — tres cortes / hasta 45 observaciones</button>
    </div>
    <div id={`${idModalidad}-panel`}>
    {!seleccion ? <p>{estadoConjunto === 'no-active' ? 'No hay un conjunto activo para comparar.' :
      'Seleccione una tienda y una familia del conjunto activo para consultar sus pronósticos.'}</p> :
      modalidad === 'corte' ? <PanelComparacion seleccion={seleccion} eleccion={eleccion} setEleccion={setEleccion} simular={simular} /> :
        <ConsolidacionEvaluaciones
          key={`${seleccion.conjuntoId}:${seleccion.serie.id}:${seleccion.serie.store_nbr}:${seleccion.serie.family}`}
          seleccion={seleccion} />}
    </div>
  </section>
}
