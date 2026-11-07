import { useId, useLayoutEffect, useRef, useState, type FormEvent } from 'react'
import type { SerieTemporal } from './SelectorSerie'

type Seleccion = { conjuntoId: number; serie: SerieTemporal }
type Props = { seleccion: Seleccion | null; activo: boolean }
const CORTES = [
  { id: '2017-06-16', nombre: '16 de junio de 2017' },
  { id: '2017-07-01', nombre: '1 de julio de 2017' },
  { id: '2017-07-16', nombre: '16 de julio de 2017' },
] as const
type Corte = typeof CORTES[number]['id']
type Resumen = { observaciones: number; total_ventas: number; promedio_por_observacion: number | null; ventas_cero: number }
type Cobertura = {
  periodo_utilizado: { inicio: string; fin: string; ambos_inclusive: true }
  primera_fecha_observada: string; ultima_fecha_observada: string
  observaciones: number; dias_naturales: number; fechas_observadas: number
  fechas_sin_registro: number; porcentaje_cobertura_calendario: number
  fechas_ausentes: string[]; definicion: string
}
type Ventas = {
  total: number; promedio_por_observacion: number; mediana: number; minimo: number; maximo: number
  desviacion_estandar_muestral: number | null; percentil_25: number; percentil_75: number
  rango_intercuartilico: number; metodo_percentiles: string; ventas_cero: number; porcentaje_cero: number
  ventas_positivas: number; ventas_fraccionarias: number; porcentaje_fraccionarias: number
}
type Estacionalidad = {
  por_dia_semana: (Resumen & { dia_semana: number })[]
  por_mes: (Resumen & { mes: number })[]
  definicion: string
}
type Promociones = {
  observaciones_sin_promocion: number; observaciones_con_promocion: number
  porcentaje_promocionadas: number; suma_conteos_onpromotion: number
  ventas_sin_promocion: Resumen; ventas_con_promocion: Resumen
}
type Extremos = {
  metodo: string; limite_inferior: number; limite_superior: number; cantidad: number
  ejemplos: { fecha: string; venta: number }[]; orden_ejemplos: string; interpretacion_iqr_cero: string
}
type Perfil = {
  alcance: 'perfil_descriptivo_serie_hasta_corte'; conjunto_id: number; serie: SerieTemporal
  fecha_corte: Corte; cobertura: Cobertura; ventas: Ventas; estacionalidad: Estacionalidad
  promociones: Promociones; valores_extremos: Extremos; advertencias: string[]
}
type Estado =
  | { kind: 'idle' }
  | { kind: 'loading'; version: number; corte: Corte }
  | { kind: 'success'; corte: Corte; datos: Perfil }
  | { kind: 'network-error' | 'http-error' | 'response-error'; corte: Corte; message: string; status?: number }

const DIA = 86_400_000
const TAMANO_PAGINA = 25
const DIAS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
const MESES = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
const numero = new Intl.NumberFormat('es-EC', { maximumFractionDigits: 3 })
const cientifico = new Intl.NumberFormat('es-EC', { notation: 'scientific', maximumFractionDigits: 3 })
const fecha = new Intl.DateTimeFormat('es-EC', { dateStyle: 'medium', timeZone: 'UTC' })

function registro(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}
function finito(v: unknown): v is number { return typeof v === 'number' && Number.isFinite(v) }
function noNegativo(v: unknown): v is number { return finito(v) && v >= 0 }
function cantidad(v: unknown): v is number { return noNegativo(v) && Number.isSafeInteger(v) }
function texto(v: unknown): v is string { return typeof v === 'string' && v.trim().length > 0 && v.length <= 2048 }
function instante(v: unknown): number | null {
  if (typeof v !== 'string' || !/^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(v) || v.startsWith('0000-')) return null
  const t = Date.parse(`${v}T00:00:00Z`)
  return Number.isFinite(t) && new Date(t).toISOString().slice(0, 10) === v ? t : null
}
function esFecha(v: unknown): v is string { return instante(v) !== null }
function esCorte(v: unknown): v is Corte { return CORTES.some(c => c.id === v) }
// Solo comprobaciones de coherencia. Nunca sustituyen los valores recibidos ni generan indicadores.
const TOLERANCIA_RELATIVA = 1e-12
const TOLERANCIA_ABSOLUTA = 1e-15
// Escalar antes de sumar evita overflow incluso con agregados próximos al máximo finito.
// La tolerancia absoluta cubre residuos cercanos a cero; la relativa conserva precisión a otras escalas.
function sumaCoherente(valores: number[], total: number): boolean {
  if (!finito(total) || !valores.every(finito)) return false
  const escala = Math.max(Math.abs(total), ...valores.map(Math.abs))
  if (escala === 0) return true
  const suma = valores.reduce((acumulado, valor) => acumulado + valor / escala, 0)
  const esperado = total / escala
  const exceso = Math.abs(suma - esperado) -
    TOLERANCIA_RELATIVA * Math.max(Math.abs(suma), Math.abs(esperado))
  // Con magnitudes subnormales, multiplicar el residuo evita desbordar la división de la tolerancia.
  return exceso <= 0 || (escala <= TOLERANCIA_ABSOLUTA
    ? exceso * escala <= TOLERANCIA_ABSOLUTA
    : exceso <= TOLERANCIA_ABSOLUTA / escala)
}
function porcentaje(v: unknown, parte: number, total: number): v is number {
  return finito(v) && v >= 0 && v <= 100 && total > 0 &&
    Math.abs(v / 100 - parte / total) <= 1e-10
}
function esResumen(v: unknown, maximo: number): v is Resumen {
  if (!registro(v) || !cantidad(v.observaciones) || v.observaciones > maximo ||
      !noNegativo(v.total_ventas) || !cantidad(v.ventas_cero) || v.ventas_cero > v.observaciones) return false
  if (v.observaciones === 0) return v.total_ventas === 0 && v.ventas_cero === 0 && v.promedio_por_observacion === null
  return noNegativo(v.promedio_por_observacion) &&
    sumaCoherente([v.total_ventas / v.observaciones], v.promedio_por_observacion) &&
    (v.ventas_cero !== v.observaciones || v.total_ventas === 0)
}
function esCobertura(v: unknown, corte: Corte): v is Cobertura {
  if (!registro(v) || !registro(v.periodo_utilizado) || !esFecha(v.primera_fecha_observada) ||
      !esFecha(v.ultima_fecha_observada) || v.primera_fecha_observada > v.ultima_fecha_observada ||
      v.ultima_fecha_observada > corte || v.periodo_utilizado.inicio !== v.primera_fecha_observada ||
      v.periodo_utilizado.fin !== corte || v.periodo_utilizado.ambos_inclusive !== true ||
      !cantidad(v.observaciones) || v.observaciones === 0 || v.fechas_observadas !== v.observaciones ||
      !cantidad(v.dias_naturales) || v.dias_naturales < v.observaciones ||
      v.fechas_sin_registro !== v.dias_naturales - v.observaciones ||
      !porcentaje(v.porcentaje_cobertura_calendario, v.observaciones, v.dias_naturales) ||
      !Array.isArray(v.fechas_ausentes) || v.fechas_ausentes.length !== v.fechas_sin_registro || !texto(v.definicion)) return false
  const inicio = instante(v.primera_fecha_observada)
  const fin = instante(corte)
  if (inicio === null || fin === null || (fin - inicio) / DIA + 1 !== v.dias_naturales) return false
  const primera = v.primera_fecha_observada
  const ultima = v.ultima_fecha_observada
  // El JSON no enumera todas las fechas observadas: estos cruces no certifican cada ausencia contra SQLite.
  return v.fechas_ausentes.every((f: unknown, i: number, todas: unknown[]) => {
    const anterior = todas[i - 1]
    return esFecha(f) && f > primera && f <= corte && f !== ultima &&
      (i === 0 || (typeof anterior === 'string' && f > anterior))
  })
}
function esVentas(v: unknown, n: number): v is Ventas {
  if (!registro(v) || !noNegativo(v.total) || !noNegativo(v.promedio_por_observacion) ||
      !noNegativo(v.mediana) || !noNegativo(v.minimo) || !noNegativo(v.maximo) ||
      !noNegativo(v.percentil_25) || !noNegativo(v.percentil_75) || !noNegativo(v.rango_intercuartilico) ||
      !texto(v.metodo_percentiles) || !cantidad(v.ventas_cero) || v.ventas_cero > n ||
      v.ventas_positivas !== n - v.ventas_cero || !cantidad(v.ventas_fraccionarias) ||
      v.ventas_fraccionarias > n - v.ventas_cero ||
      !porcentaje(v.porcentaje_cero, v.ventas_cero, n) ||
      !porcentaje(v.porcentaje_fraccionarias, v.ventas_fraccionarias, n)) return false
  return v.minimo <= v.percentil_25 && v.percentil_25 <= v.mediana && v.mediana <= v.percentil_75 &&
    sumaCoherente([v.total / n], v.promedio_por_observacion) &&
    sumaCoherente([v.percentil_75 - v.percentil_25], v.rango_intercuartilico) &&
    v.percentil_75 <= v.maximo && v.total >= v.maximo &&
    (v.promedio_por_observacion >= v.minimo || sumaCoherente([v.minimo], v.promedio_por_observacion)) &&
    (v.promedio_por_observacion <= v.maximo || sumaCoherente([v.maximo], v.promedio_por_observacion)) &&
    v.rango_intercuartilico <= v.maximo &&
    (n === 1 ? v.desviacion_estandar_muestral === null : noNegativo(v.desviacion_estandar_muestral)) &&
    (v.ventas_cero !== n || v.maximo === 0)
}
function esEstacionalidad(v: unknown, n: number, ventas: Ventas): v is Estacionalidad {
  if (!registro(v) || !texto(v.definicion)) return false
  function grupos(a: unknown, campo: 'dia_semana' | 'mes', minimo: number, maximo: number): boolean {
    if (!Array.isArray(a) || a.length === 0 || a.length > maximo - minimo + 1) return false
    let anterior = minimo - 1
    let observaciones = 0
    let ceros = 0
    const totales: number[] = []
    for (const item of a as unknown[]) {
      if (!registro(item)) return false
      const clave = item[campo]
      if (!esResumen(item, n) || item.observaciones === 0 || !cantidad(clave) ||
          clave < minimo || clave > maximo || clave <= anterior) return false
      anterior = clave
      observaciones += item.observaciones
      ceros += item.ventas_cero
      totales.push(item.total_ventas)
    }
    return observaciones === n && ceros === ventas.ventas_cero && sumaCoherente(totales, ventas.total)
  }
  return grupos(v.por_dia_semana, 'dia_semana', 0, 6) && grupos(v.por_mes, 'mes', 1, 12)
}
function esPromociones(v: unknown, n: number, ventas: Ventas): v is Promociones {
  return registro(v) && cantidad(v.observaciones_sin_promocion) && cantidad(v.observaciones_con_promocion) &&
    v.observaciones_sin_promocion + v.observaciones_con_promocion === n &&
    porcentaje(v.porcentaje_promocionadas, v.observaciones_con_promocion, n) &&
    cantidad(v.suma_conteos_onpromotion) && v.suma_conteos_onpromotion >= v.observaciones_con_promocion &&
    esResumen(v.ventas_sin_promocion, n) && v.ventas_sin_promocion.observaciones === v.observaciones_sin_promocion &&
    esResumen(v.ventas_con_promocion, n) && v.ventas_con_promocion.observaciones === v.observaciones_con_promocion &&
    v.ventas_sin_promocion.ventas_cero + v.ventas_con_promocion.ventas_cero === ventas.ventas_cero &&
    sumaCoherente([v.ventas_sin_promocion.total_ventas, v.ventas_con_promocion.total_ventas], ventas.total)
}
function esExtremos(v: unknown, c: Cobertura, ventas: Ventas): v is Extremos {
  if (!registro(v) || !texto(v.metodo) || !texto(v.orden_ejemplos) || !texto(v.interpretacion_iqr_cero) ||
      !finito(v.limite_inferior) || !finito(v.limite_superior) || v.limite_inferior > v.limite_superior ||
      !cantidad(v.cantidad) || v.cantidad > c.observaciones || !Array.isArray(v.ejemplos) ||
      v.ejemplos.length !== Math.min(10, v.cantidad)) return false
  if (ventas.rango_intercuartilico > Number.MAX_VALUE / 1.5) return false
  const margen = 1.5 * ventas.rango_intercuartilico
  if (!sumaCoherente([ventas.percentil_25, -margen], v.limite_inferior) ||
      !sumaCoherente([ventas.percentil_75, margen], v.limite_superior)) return false
  const ausentes = new Set(c.fechas_ausentes)
  let anterior = ''
  for (const item of v.ejemplos as unknown[]) {
    if (!registro(item) || !esFecha(item.fecha) || item.fecha <= anterior || ausentes.has(item.fecha) ||
        item.fecha < c.primera_fecha_observada || item.fecha > c.ultima_fecha_observada ||
        !noNegativo(item.venta) || item.venta < ventas.minimo || item.venta > ventas.maximo ||
        !(item.venta < v.limite_inferior || item.venta > v.limite_superior)) return false
    anterior = item.fecha
  }
  return true
}
function esPerfil(v: unknown, seleccion: Seleccion, corte: Corte): v is Perfil {
  if (!registro(v) || v.alcance !== 'perfil_descriptivo_serie_hasta_corte' ||
      !cantidad(v.conjunto_id) || v.conjunto_id === 0 || v.conjunto_id !== seleccion.conjuntoId || !registro(v.serie) ||
      !cantidad(v.serie.id) || v.serie.id === 0 || !cantidad(v.serie.store_nbr) || v.serie.store_nbr === 0 || !texto(v.serie.family) ||
      v.serie.id !== seleccion.serie.id || v.serie.store_nbr !== seleccion.serie.store_nbr ||
      v.serie.family !== seleccion.serie.family || v.fecha_corte !== corte || !esCobertura(v.cobertura, corte)) return false
  const n = v.cobertura.observaciones
  return esVentas(v.ventas, n) && esEstacionalidad(v.estacionalidad, n, v.ventas) && esPromociones(v.promociones, n, v.ventas) &&
    esExtremos(v.valores_extremos, v.cobertura, v.ventas) && Array.isArray(v.advertencias) &&
    v.advertencias.length <= 50 && v.advertencias.every(texto)
}

function presentar(v: number | null): string {
  if (v === null) return 'No calculable'
  return v !== 0 && (Math.abs(v) < 0.001 || Math.abs(v) >= 1e9) ? cientifico.format(v) : numero.format(v)
}
function Fecha({ valor }: { valor: string }) {
  return <time dateTime={valor}>{fecha.format(new Date(`${valor}T00:00:00Z`))}</time>
}
function Indicador({ nombre, valor, sufijo = '' }: { nombre: string; valor: number | null; sufijo?: string }) {
  return <div><dt>{nombre}</dt><dd title={valor === null ? 'No definido para estas observaciones' : String(valor)}>
    {presentar(valor)}{valor === null ? '' : sufijo}</dd></div>
}
function Barras({ grupos, etiqueta, titulo }: { grupos: (Resumen & { etiqueta: string })[]; etiqueta: string; titulo: string }) {
  const maximo = Math.max(0, ...grupos.map(g => g.total_ventas))
  return <figure className="profile-bars" aria-label={etiqueta}>
    <figcaption><strong>{titulo}</strong><span>Ventas totales observadas. Escala lineal desde cero hasta {presentar(maximo)}.</span></figcaption>
    {maximo === 0 && <p>Todos los totales observados son cero.</p>}
    <ul>{grupos.map(g => <li key={g.etiqueta}>
      <div className="profile-bar-label"><strong>{g.etiqueta}</strong><span>Total: {presentar(g.total_ventas)}</span></div>
      <div className="profile-bar-track" aria-hidden="true"><span style={{ width: `${maximo > 0 ? g.total_ventas / maximo * 100 : 0}%` }} /></div>
      <p className="help-text">{presentar(g.observaciones)} observaciones · Promedio: {presentar(g.promedio_por_observacion)} · Ceros: {presentar(g.ventas_cero)}</p>
    </li>)}</ul>
    <p className="help-text">Solo se presentan grupos con observaciones; no se completan grupos ausentes.</p>
  </figure>
}
function mensajeHTTP(status: number, body: unknown): string {
  // detail puede ser string o un arreglo de FastAPI. No se muestran objetos ni texto técnico recibido.
  if (status === 422 && registro(body) && Array.isArray(body.detail)) {
    return 'Revise la serie y seleccione uno de los tres cortes autorizados.'
  }
  if (registro(body) && typeof body.detail === 'string' && status === 422) {
    return 'El corte o la serie no son válidos para este análisis. Revise la selección.'
  }
  const mensajes: Record<number, string> = {
    404: 'La serie no está disponible en el conjunto activo. Recargue la página para actualizar las series.',
    409: 'No se puede analizar este histórico. Compruebe que hay datos válidos y un conjunto activo.',
    422: 'Revise la serie y el corte de análisis.',
    503: 'La base de datos no está disponible. Inténtelo nuevamente más tarde.',
    500: 'El servicio no pudo completar el análisis. Inténtelo nuevamente.',
  }
  return mensajes[status] ?? 'El servicio no pudo atender la consulta del perfil.'
}

function PanelPerfil({ seleccion, activo }: { seleccion: Seleccion; activo: boolean }) {
  const id = useId()
  const [corte, setCorte] = useState<Corte>('2017-06-16')
  const [estado, setEstado] = useState<Estado>({ kind: 'idle' })
  const [pagina, setPagina] = useState(1)
  const solicitud = useRef<AbortController | null>(null)
  const revision = useRef(0)
  const contexto = JSON.stringify([seleccion.conjuntoId, seleccion.serie.id, seleccion.serie.store_nbr, seleccion.serie.family])

  // El cleanup invalida la identidad antes de abortar. Una red que ignore abort no puede publicar su respuesta.
  // Un resultado ya terminado permanece en estado; el contenedor lo oculta mientras la vista está inactiva.
  useLayoutEffect(() => () => {
    revision.current += 1
    const pendiente = solicitud.current
    solicitud.current = null
    pendiente?.abort()
  }, [activo, contexto, corte])

  function cambiarCorte(valor: string) {
    if (!esCorte(valor) || valor === corte) return
    revision.current += 1
    const pendiente = solicitud.current
    solicitud.current = null
    pendiente?.abort()
    setCorte(valor)
    setPagina(1)
    setEstado({ kind: 'idle' })
  }

  async function analizar(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!activo || solicitud.current !== null) return
    const controller = new AbortController()
    const version = ++revision.current
    solicitud.current = controller  // exclusión inmediata, incluso antes del siguiente render/doble clic
    const esperado = { conjuntoId: seleccion.conjuntoId, serie: { ...seleccion.serie } }
    const solicitado = corte
    const actual = () => revision.current === version && solicitud.current === controller && !controller.signal.aborted
    controller.signal.addEventListener('abort', () => {
      setEstado(anterior => anterior.kind === 'loading' && anterior.version === version ? { kind: 'idle' } : anterior)
    }, { once: true })
    setPagina(1)
    setEstado({ kind: 'loading', version, corte: solicitado })
    try {
      let response: Response
      try {
        response = await fetch(`/api/datasets/active/series/${esperado.serie.id}/perfil?${new URLSearchParams({ fecha_corte: solicitado })}`, { signal: controller.signal })
      } catch {
        if (actual()) setEstado({ kind: 'network-error', corte: solicitado, message: 'No se pudo conectar con el servicio de perfiles. Compruebe la conexión e inténtelo nuevamente.' })
        return
      }
      if (!actual()) return
      let body: unknown
      try { body = await response.json() } catch {
        if (actual()) setEstado(response.ok
          ? { kind: 'response-error', corte: solicitado, message: 'El servicio no devolvió un JSON de perfil válido.' }
          : { kind: 'http-error', corte: solicitado, status: response.status, message: mensajeHTTP(response.status, null) })
        return
      }
      if (!actual()) return
      if (!response.ok) {
        setEstado({ kind: 'http-error', corte: solicitado, status: response.status, message: mensajeHTTP(response.status, body) })
      } else if (!esPerfil(body, esperado, solicitado)) {
        setEstado({ kind: 'response-error', corte: solicitado, message: 'La respuesta es incompatible con el perfil de esta serie y corte. No se muestran sus datos; actualice la selección y vuelva a consultar.' })
      } else {
        setEstado({ kind: 'success', corte: solicitado, datos: body })
      }
    } finally {
      if (solicitud.current === controller) solicitud.current = null
    }
  }

  const vigente = estado.kind === 'idle' || estado.corte === corte ? estado : { kind: 'idle' as const }
  const cargando = vigente.kind === 'loading' && activo
  const datos = vigente.kind === 'success' && activo ? vigente.datos : null
  const error = vigente.kind === 'network-error' || vigente.kind === 'http-error' || vigente.kind === 'response-error' ? vigente : null
  const paginas = datos ? Math.max(1, Math.ceil(datos.cobertura.fechas_ausentes.length / TAMANO_PAGINA)) : 1

  return <section className="profile-card" aria-labelledby={`${id}-heading`}>
    <h2 id={`${id}-heading`}>Perfil descriptivo de la serie</h2>
    <p className="profile-context">Conjunto {seleccion.conjuntoId} · <strong>Tienda {seleccion.serie.store_nbr} · {seleccion.serie.family}</strong></p>
    <form className="profile-form" onSubmit={event => { void analizar(event) }} aria-busy={cargando}>
      <div><label htmlFor={`${id}-cut`}>Corte del perfil</label>
        <select id={`${id}-cut`} value={corte} onChange={event => cambiarCorte(event.target.value)} aria-describedby={`${id}-help`}>
          {CORTES.map(c => <option key={c.id} value={c.id}>{c.nombre}</option>)}
        </select></div>
      <button type="submit" disabled={cargando || !activo}>Analizar serie</button>
      <p className="help-text" id={`${id}-help`}>Consulta descriptiva hasta el corte. No entrena modelos ni genera pronósticos.</p>
    </form>
    {vigente.kind === 'idle' && <p className="profile-message">Listo para consultar. El perfil se solicita únicamente al pulsar «Analizar serie».</p>}
    {cargando && <p className="profile-message" role="status" aria-live="polite">Consultando perfil…</p>}
    {error && activo && <div className="profile-message profile-error" role="alert">
      <strong>{error.kind === 'network-error' ? 'Error de conexión' : error.kind === 'response-error' ? 'Respuesta incompatible' : `Error HTTP ${error.status}`}</strong>
      <p>{error.message}</p>
    </div>}
    {datos && <div className="profile-results">
      <p className="profile-period">Perfil hasta <strong><Fecha valor={datos.fecha_corte} /></strong>. Periodo observado desde <Fecha valor={datos.cobertura.primera_fecha_observada} />.</p>
      <dl className="profile-metrics">
        <Indicador nombre="Total de ventas" valor={datos.ventas.total} />
        <Indicador nombre="Promedio por observación" valor={datos.ventas.promedio_por_observacion} />
        <Indicador nombre="Mediana" valor={datos.ventas.mediana} />
        <Indicador nombre="Observaciones" valor={datos.cobertura.observaciones} />
        <Indicador nombre="Días sin registro" valor={datos.cobertura.fechas_sin_registro} />
        <Indicador nombre="Observaciones con ventas cero" valor={datos.ventas.ventas_cero} />
      </dl>
      <p className="profile-note">Un día sin registro no es una venta cero. No se completan ni interpolan observaciones.</p>

      <details className="profile-section"><summary>Cobertura del calendario</summary>
        <dl className="profile-metrics">
          <div><dt>Primera fecha observada</dt><dd><Fecha valor={datos.cobertura.primera_fecha_observada} /></dd></div>
          <div><dt>Última fecha observada</dt><dd><Fecha valor={datos.cobertura.ultima_fecha_observada} /></dd></div>
          <div><dt>Fecha de corte</dt><dd><Fecha valor={datos.fecha_corte} /></dd></div>
          <Indicador nombre="Días naturales del periodo" valor={datos.cobertura.dias_naturales} />
          <Indicador nombre="Fechas observadas" valor={datos.cobertura.fechas_observadas} />
          <Indicador nombre="Cobertura calendario" valor={datos.cobertura.porcentaje_cobertura_calendario} sufijo=" %" />
        </dl><p className="help-text">{datos.cobertura.definicion}</p>
      </details>

      <details className="profile-section"><summary>Estadísticas de ventas</summary>
        <dl className="profile-metrics">
          <Indicador nombre="Mínimo" valor={datos.ventas.minimo} /><Indicador nombre="Máximo" valor={datos.ventas.maximo} />
          <Indicador nombre="Desviación estándar muestral" valor={datos.ventas.desviacion_estandar_muestral} />
          <Indicador nombre="Percentil 25" valor={datos.ventas.percentil_25} /><Indicador nombre="Percentil 75" valor={datos.ventas.percentil_75} />
          <Indicador nombre="Rango intercuartílico" valor={datos.ventas.rango_intercuartilico} />
          <Indicador nombre="Ventas positivas" valor={datos.ventas.ventas_positivas} />
          <Indicador nombre="Ventas fraccionarias" valor={datos.ventas.ventas_fraccionarias} />
          <Indicador nombre="Porcentaje de ventas fraccionarias" valor={datos.ventas.porcentaje_fraccionarias} sufijo=" %" />
          <Indicador nombre="Porcentaje de ventas cero" valor={datos.ventas.porcentaje_cero} sufijo=" %" />
        </dl><p className="help-text">{datos.ventas.metodo_percentiles} La desviación muestral no es calculable con una sola observación.</p>
      </details>

      <details className="profile-section"><summary>Distribución por día de semana y mes</summary>
        <p className="profile-note">Las diferencias descriptivas no demuestran estacionalidad significativa.</p>
        <Barras titulo="Por día de semana" etiqueta="Ventas totales por día de semana, con valores numéricos" grupos={datos.estacionalidad.por_dia_semana.map(g => ({ ...g, etiqueta: DIAS[g.dia_semana] }))} />
        <Barras titulo="Por mes del año" etiqueta="Ventas totales por mes del año, con valores numéricos" grupos={datos.estacionalidad.por_mes.map(g => ({ ...g, etiqueta: MESES[g.mes - 1] }))} />
      </details>

      <details className="profile-section"><summary>Promociones · descripción de observaciones</summary>
        <dl className="profile-metrics">
          <Indicador nombre="Observaciones sin promoción" valor={datos.promociones.observaciones_sin_promocion} />
          <Indicador nombre="Observaciones con promoción" valor={datos.promociones.observaciones_con_promocion} />
          <Indicador nombre="Observaciones promocionadas" valor={datos.promociones.porcentaje_promocionadas} sufijo=" %" />
          <Indicador nombre="Suma literal de conteos onpromotion" valor={datos.promociones.suma_conteos_onpromotion} />
        </dl>
        <div className="profile-table-wrap" tabIndex={0} role="region" aria-label="Ventas según promoción, tabla desplazable">
          <table><caption>Ventas observadas con y sin promoción</caption><thead><tr><th scope="col">Observaciones</th><th scope="col">Cantidad</th><th scope="col">Total de ventas</th><th scope="col">Promedio</th></tr></thead>
            <tbody>{[{ nombre: 'Sin promoción', grupo: datos.promociones.ventas_sin_promocion },
              { nombre: 'Con promoción', grupo: datos.promociones.ventas_con_promocion }].map(({ nombre, grupo }) =>
              <tr key={nombre}><th scope="row">{nombre}</th><td>{presentar(grupo.observaciones)}</td><td>{presentar(grupo.total_ventas)}</td><td>{presentar(grupo.promedio_por_observacion)}</td></tr>)}</tbody>
          </table>
        </div><p className="profile-note">Análisis descriptivo: no prueba causalidad entre promociones y ventas. Los conteos no representan artículos únicos.</p>
      </details>

      <details className="profile-section"><summary>Valores fuera de límites IQR</summary>
        <p>{datos.valores_extremos.metodo}</p>
        <dl className="profile-metrics"><Indicador nombre="Límite inferior descriptivo" valor={datos.valores_extremos.limite_inferior} />
          <Indicador nombre="Límite superior descriptivo" valor={datos.valores_extremos.limite_superior} />
          <Indicador nombre="Observaciones fuera del intervalo" valor={datos.valores_extremos.cantidad} /></dl>
        {datos.ventas.rango_intercuartilico === 0 && <p>{datos.valores_extremos.interpretacion_iqr_cero}</p>}
        {datos.valores_extremos.cantidad === 0 ? <p>Ninguna observación fuera del intervalo descriptivo.</p> : <div className="profile-table-wrap" tabIndex={0} role="region" aria-label="Ejemplos fuera del intervalo, tabla desplazable">
          <table><caption>Hasta diez ejemplos, ordenados por fecha</caption><thead><tr><th scope="col">Fecha</th><th scope="col">Venta observada</th></tr></thead>
            <tbody>{datos.valores_extremos.ejemplos.map(e => <tr key={e.fecha}><td><Fecha valor={e.fecha} /></td><td>{presentar(e.venta)}</td></tr>)}</tbody>
          </table></div>}
        <p className="profile-note">Superar un límite IQR no confirma una anomalía. No se eliminan ni corrigen observaciones.</p>
      </details>

      <details className="profile-section"><summary>Consultar fechas sin registro · {presentar(datos.cobertura.fechas_sin_registro)}</summary>
        {datos.cobertura.fechas_sin_registro === 0 ? <p>No hay fechas sin registro dentro del periodo utilizado.</p> : <>
          <p>Fechas ausentes; no se les asigna una venta de cero.</p>
          <ul className="profile-missing">{datos.cobertura.fechas_ausentes.slice((pagina - 1) * TAMANO_PAGINA, pagina * TAMANO_PAGINA).map(f => <li key={f}><Fecha valor={f} /></li>)}</ul>
          <nav className="profile-pagination" aria-label="Paginación de fechas ausentes">
            <button type="button" disabled={pagina === 1} onClick={() => setPagina(p => Math.max(1, p - 1))}>Anterior</button>
            <span role="status">Página {presentar(pagina)} de {presentar(paginas)} · Hasta 25 fechas por página</span>
            <button type="button" disabled={pagina === paginas} onClick={() => setPagina(p => Math.min(paginas, p + 1))}>Siguiente</button>
          </nav>
        </>}
      </details>
      <details className="profile-section"><summary>Alcance y advertencias</summary><ul>{datos.advertencias.map((a, i) => <li key={i}>{a}</li>)}</ul></details>
    </div>}
  </section>
}

function PerfilSerie({ seleccion, activo }: Props) {
  if (seleccion === null) return <section className="profile-card" aria-label="Perfil descriptivo de la serie">
    <h2>Perfil descriptivo de la serie</h2><p>Seleccione una tienda y una familia para consultar su perfil.</p>
  </section>
  return <PanelPerfil key={JSON.stringify([seleccion.conjuntoId, seleccion.serie.id, seleccion.serie.store_nbr, seleccion.serie.family])} seleccion={seleccion} activo={activo} />
}

export default PerfilSerie
