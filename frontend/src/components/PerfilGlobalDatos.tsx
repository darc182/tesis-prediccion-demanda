import { useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import type { SeriesState } from './SelectorSerie'

type ConteoFecha = { fecha: string; filas: number }
type Tienda = { store_nbr: number; filas: number }
type Familia = { family: string; filas: number; ventas: number }
type Combinacion = Familia & { store_nbr: number }
type Ventas = {
  suma: number; promedio_por_registro: number; minimo: number; maximo: number
  registros_cero: number; porcentaje_cero: number; registros_fraccionarios: number
  porcentaje_fraccionarios: number; registros_positivos: number
}
type Promociones = {
  registros_cero: number; registros_positivos: number; porcentaje_con_promocion: number
  suma_conteos_onpromotion: number; minimo: number; maximo: number; promedio_por_registro: number
}
type Reporte = {
  fuente: { nombre_archivo: string; columnas_examinadas: string[] }
  alcance: 'panorama_descriptivo_global'
  interpretacion_temporal: { retrospectivo: true; incluye_todas_las_fechas_del_archivo: true; autorizado_para_seleccion_de_modelos: false; nota: string }
  volumen: { filas_examinadas: number; filas_validas: number; filas_invalidas: number; tiendas_distintas: number; familias_distintas: number; combinaciones_observadas: number }
  temporalidad: { primera_fecha: string; ultima_fecha: string; fechas_distintas: number; dias_calendario_inclusivos: number; fechas_sin_registro_global: string[]; filas_por_fecha: ConteoFecha[]; filas_por_anio: { anio: number; filas: number }[] }
  ventas: Ventas
  promociones: Promociones
  distribuciones: { por_tienda: Tienda[]; por_familia: Familia[]; combinaciones_mayor_volumen: Combinacion[]; criterio_mayores: string }
  advertencias: string[]
}
type PerfilGlobal = { conjunto_id: number; version_perfil: 1; sha256_csv: string; series_importadas: number; observaciones_importadas: number; reporte: Reporte }
type Props = { conjuntoIdActivo: number | null; estadoConjunto: SeriesState['kind']; activo: boolean; actualizarSeries: () => void }
type Estado = { kind: 'idle' } | { kind: 'loading'; signal: AbortSignal } | { kind: 'success'; datos: PerfilGlobal } |
  { kind: 'error'; titulo: string; mensaje: string; actualizar?: boolean }

const DIA = 86_400_000
const PAGINA = 25
const formato = new Intl.NumberFormat('es-EC', { maximumSignificantDigits: 12 })
const cientifico = new Intl.NumberFormat('es-EC', { notation: 'scientific', maximumSignificantDigits: 8 })
const fechaLocal = new Intl.DateTimeFormat('es-EC', { dateStyle: 'medium', timeZone: 'UTC' })
function numero(n: number): string { return (n !== 0 && (Math.abs(n) < 1e-6 || Math.abs(n) >= 1e15) ? cientifico : formato).format(n) }
function fecha(f: string): string { return fechaLocal.format(new Date(`${f}T00:00:00Z`)) }
function objeto(v: unknown): v is Record<string, unknown> { return typeof v === 'object' && v !== null && !Array.isArray(v) }
function finito(v: unknown): v is number { return typeof v === 'number' && Number.isFinite(v) && v >= 0 }
function entero(v: unknown): v is number { return finito(v) && Number.isSafeInteger(v) }
function positivo(v: unknown): v is number { return entero(v) && v > 0 }
function texto(v: unknown): v is string { return typeof v === 'string' && v.trim().length > 0 && v.length <= 4096 && Array.from(v).every(c => c.charCodeAt(0) >= 32 && c.charCodeAt(0) !== 127) }
// G1/G5-B1 cuentan puntos de código y admiten saltos/tabuladores en familias CSV.
function familiaValida(v: unknown): v is string { return typeof v === 'string' && v.trim().length > 0 && Array.from(v).length <= 512 }
function canonica(v: unknown): v is string {
  if (typeof v !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(v) || v.startsWith('0000')) return false
  const n = Date.parse(`${v}T00:00:00Z`)
  return Number.isFinite(n) && new Date(n).toISOString().slice(0, 10) === v
}
// Validación de agregados, nunca sustitución de los valores recibidos.
// rtol=1e-12, atol=1e-15; suma compensada y escalada para evitar overflow.
function coherente(valores: number[], total: number): boolean {
  if (!finito(total) || !valores.every(finito)) return false
  let escala = total
  for (const v of valores) escala = Math.max(escala, v)
  if (escala === 0) return true
  let suma = 0, compensacion = 0
  for (const v of valores) { const y = v / escala - compensacion; const t = suma + y; compensacion = (t - suma) - y; suma = t }
  const esperado = total / escala
  const exceso = Math.abs(suma - esperado) - 1e-12 * Math.max(suma, esperado)
  return exceso <= 0 || (escala <= 1e-15 ? exceso * escala <= 1e-15 : exceso <= 1e-15 / escala)
}
function porcentaje(v: unknown, parte: number, total: number): v is number {
  return finito(v) && v <= 100 && coherente([parte / total], v / 100)
}
function lista<T>(v: unknown, guard: (x: unknown) => x is T, max: number): v is T[] {
  return Array.isArray(v) && v.length <= max && v.every(guard)
}
function ventasValidas(v: unknown, n: number): v is Ventas {
  if (!objeto(v) || !finito(v.suma) || !finito(v.promedio_por_registro) || !finito(v.minimo) || !finito(v.maximo) ||
      !entero(v.registros_cero) || !entero(v.registros_positivos) || !entero(v.registros_fraccionarios)) return false
  return v.registros_cero <= n && v.registros_positivos === n - v.registros_cero && v.registros_fraccionarios <= v.registros_positivos &&
    (v.suma > 0) === (v.registros_positivos > 0) &&
    v.minimo <= v.maximo && v.maximo <= v.suma && (v.minimo === 0) === (v.registros_cero > 0) && (v.maximo > 0) === (v.registros_positivos > 0) &&
    coherente([v.suma / n], v.promedio_por_registro) &&
    (v.promedio_por_registro >= v.minimo || coherente([v.minimo], v.promedio_por_registro)) &&
    (v.promedio_por_registro <= v.maximo || coherente([v.maximo], v.promedio_por_registro)) &&
    porcentaje(v.porcentaje_cero, v.registros_cero, n) && porcentaje(v.porcentaje_fraccionarios, v.registros_fraccionarios, n)
}
function promocionesValidas(p: unknown, n: number): p is Promociones {
  if (!objeto(p) || !entero(p.registros_cero) || !entero(p.registros_positivos) || !entero(p.suma_conteos_onpromotion) ||
      !entero(p.minimo) || !entero(p.maximo) || !finito(p.promedio_por_registro)) return false
  return p.registros_cero <= n && p.registros_positivos === n - p.registros_cero && p.registros_positivos <= p.suma_conteos_onpromotion &&
    (p.suma_conteos_onpromotion > 0) === (p.registros_positivos > 0) &&
    p.minimo <= p.maximo && (p.minimo === 0) === (p.registros_cero > 0) && (p.maximo > 0) === (p.registros_positivos > 0) &&
    p.maximo <= p.suma_conteos_onpromotion && coherente([p.suma_conteos_onpromotion / n], p.promedio_por_registro) &&
    (p.promedio_por_registro >= p.minimo || coherente([p.minimo], p.promedio_por_registro)) &&
    (p.promedio_por_registro <= p.maximo || coherente([p.maximo], p.promedio_por_registro)) &&
    porcentaje(p.porcentaje_con_promocion, p.registros_positivos, n)
}
function calendarioValido(t: unknown, n: number): t is Reporte['temporalidad'] {
  if (!objeto(t) || !canonica(t.primera_fecha) || !canonica(t.ultima_fecha) || t.primera_fecha > t.ultima_fecha ||
      !positivo(t.fechas_distintas) || !positivo(t.dias_calendario_inclusivos) || t.dias_calendario_inclusivos > 36600 || t.fechas_distintas > n ||
      !lista(t.fechas_sin_registro_global, canonica, 36600) || !lista(t.filas_por_fecha,
        (v): v is ConteoFecha => objeto(v) && canonica(v.fecha) && positivo(v.filas) && v.filas <= n, 36600) ||
      !lista(t.filas_por_anio, (v): v is { anio: number; filas: number } => objeto(v) && positivo(v.anio) && v.anio <= 9999 && positivo(v.filas) && v.filas <= n, 36600)) return false
  const inicio = t.primera_fecha, fin = t.ultima_fecha
  if ((Date.parse(fin) - Date.parse(inicio)) / DIA + 1 !== t.dias_calendario_inclusivos || t.filas_por_fecha.length !== t.fechas_distintas ||
      t.fechas_sin_registro_global.length + t.fechas_distintas !== t.dias_calendario_inclusivos) return false
  const anios = new Map<number, number>()
  const observadas = new Set<string>()
  let filas = 0
  for (const [i, f] of t.filas_por_fecha.entries()) {
    if (f.fecha < inicio || f.fecha > fin || (i > 0 && f.fecha <= t.filas_por_fecha[i - 1].fecha)) return false
    filas += f.filas
    if (!Number.isSafeInteger(filas) || filas > n) return false
    observadas.add(f.fecha)
    const a = Number(f.fecha.slice(0, 4)); anios.set(a, (anios.get(a) ?? 0) + f.filas)
  }
  return filas === n && t.filas_por_fecha[0]?.fecha === inicio && t.filas_por_fecha.at(-1)?.fecha === fin &&
    t.fechas_sin_registro_global.every((f, i, a) => f >= inicio && f <= fin && !observadas.has(f) && (i === 0 || f > a[i - 1])) &&
    t.filas_por_anio.length === anios.size && t.filas_por_anio.every((a, i, todos) => anios.get(a.anio) === a.filas && (i === 0 || a.anio > todos[i - 1].anio))
}
function distribucionesValidas(d: unknown, v: Reporte['volumen'], total: number): d is Reporte['distribuciones'] {
  if (!objeto(d) || !texto(d.criterio_mayores) ||
      !lista(d.por_tienda, (x): x is Tienda => objeto(x) && positivo(x.store_nbr) && positivo(x.filas) && x.filas <= v.filas_validas, 1000) ||
      !lista(d.por_familia, (x): x is Familia => objeto(x) && familiaValida(x.family) && positivo(x.filas) && x.filas <= v.filas_validas && finito(x.ventas), 1000) ||
      !lista(d.combinaciones_mayor_volumen, (x): x is Combinacion => objeto(x) && positivo(x.store_nbr) && familiaValida(x.family) && positivo(x.filas) && finito(x.ventas), 10)) return false
  if (d.por_tienda.length !== v.tiendas_distintas || d.por_familia.length !== v.familias_distintas ||
      d.combinaciones_mayor_volumen.length !== Math.min(10, v.combinaciones_observadas) ||
      !d.por_tienda.every((x, i, a) => i === 0 || x.store_nbr > a[i - 1].store_nbr) ||
      !d.por_familia.every((x, i, a) => i === 0 || compararTexto(x.family, a[i - 1].family) > 0)) return false
  const n = v.filas_validas
  if (d.por_tienda.reduce((s, x) => s + x.filas, 0) !== n || d.por_familia.reduce((s, x) => s + x.filas, 0) !== n ||
      !coherente(d.por_familia.map(x => x.ventas), total)) return false
  const claves = new Set<string>()
  const filasT = new Map<number, number>(), filasF = new Map<string, number>(), ventasF = new Map<string, number[]>()
  for (const [i, x] of d.combinaciones_mayor_volumen.entries()) {
    const anterior = d.combinaciones_mayor_volumen[i - 1]
    const clave = JSON.stringify([x.store_nbr, x.family])
    if (claves.has(clave) || !d.por_tienda.some(t => t.store_nbr === x.store_nbr) || !d.por_familia.some(f => f.family === x.family) ||
        (anterior && (x.ventas > anterior.ventas || (x.ventas === anterior.ventas &&
          (x.store_nbr < anterior.store_nbr || (x.store_nbr === anterior.store_nbr && compararTexto(x.family, anterior.family) <= 0)))))) return false
    claves.add(clave); filasT.set(x.store_nbr, (filasT.get(x.store_nbr) ?? 0) + x.filas)
    filasF.set(x.family, (filasF.get(x.family) ?? 0) + x.filas)
    ventasF.set(x.family, [...(ventasF.get(x.family) ?? []), x.ventas])
  }
  return d.por_tienda.every(t => (filasT.get(t.store_nbr) ?? 0) <= t.filas && (v.combinaciones_observadas > 10 || filasT.get(t.store_nbr) === t.filas)) &&
    d.por_familia.every(f => {
      const montos = ventasF.get(f.family) ?? []
      const escala = Math.max(f.ventas, ...montos)
      const acotada = escala === 0 || montos.reduce((s, x) => s + x / escala, 0) <= f.ventas / escala || coherente(montos, f.ventas)
      return (filasF.get(f.family) ?? 0) <= f.filas && acotada && (v.combinaciones_observadas > 10 ||
        (filasF.get(f.family) === f.filas && coherente(montos, f.ventas)))
    })
}
// Python ordena por puntos Unicode, no por unidades UTF-16 ni por reglas locales.
function compararTexto(a: string, b: string): number {
  const aa = Array.from(a, c => c.codePointAt(0) ?? 0), bb = Array.from(b, c => c.codePointAt(0) ?? 0)
  for (let i = 0; i < Math.min(aa.length, bb.length); i++) if (aa[i] !== bb[i]) return aa[i] - bb[i]
  return aa.length - bb.length
}
function esPerfilGlobal(p: unknown): p is PerfilGlobal {
  if (!objeto(p) || !positivo(p.conjunto_id) || p.version_perfil !== 1 || typeof p.sha256_csv !== 'string' || !/^[0-9a-f]{64}$/.test(p.sha256_csv) ||
      !entero(p.series_importadas) || !entero(p.observaciones_importadas) || !objeto(p.reporte)) return false
  const r = p.reporte, v = r.volumen, f = r.fuente, i = r.interpretacion_temporal
  if (!objeto(f) || !texto(f.nombre_archivo) || /[/\\:]/.test(f.nombre_archivo) || !/\.csv$/i.test(f.nombre_archivo) ||
      !Array.isArray(f.columnas_examinadas) || JSON.stringify(f.columnas_examinadas) !== JSON.stringify(['date', 'store_nbr', 'family', 'sales', 'onpromotion']) ||
      r.alcance !== 'panorama_descriptivo_global' || !objeto(i) || i.retrospectivo !== true || i.incluye_todas_las_fechas_del_archivo !== true ||
      i.autorizado_para_seleccion_de_modelos !== false || !texto(i.nota) || !objeto(v) || !positivo(v.filas_examinadas) ||
      v.filas_validas !== v.filas_examinadas || v.filas_invalidas !== 0 || !positivo(v.tiendas_distintas) || !positivo(v.familias_distintas) || !positivo(v.combinaciones_observadas)) return false
  const volumen: Reporte['volumen'] = { filas_examinadas: v.filas_examinadas, filas_validas: v.filas_validas, filas_invalidas: 0,
    tiendas_distintas: v.tiendas_distintas, familias_distintas: v.familias_distintas, combinaciones_observadas: v.combinaciones_observadas }
  return v.tiendas_distintas <= v.combinaciones_observadas && v.familias_distintas <= v.combinaciones_observadas &&
    v.combinaciones_observadas <= Math.min(v.filas_validas, 10000, v.tiendas_distintas * v.familias_distintas) &&
    p.series_importadas <= v.combinaciones_observadas && p.observaciones_importadas <= v.filas_validas &&
    p.series_importadas <= p.observaciones_importadas && (p.series_importadas === 0) === (p.observaciones_importadas === 0) &&
    ventasValidas(r.ventas, v.filas_validas) && promocionesValidas(r.promociones, v.filas_validas) && calendarioValido(r.temporalidad, v.filas_validas) &&
    distribucionesValidas(r.distribuciones, volumen, r.ventas.suma) && lista(r.advertencias, texto, 16) && r.advertencias.length > 0
}

function Indicadores({ items }: { items: [string, number | string][] }) {
  return <dl className="global-profile-metrics">{items.map(([label, value]) => <div key={label}><dt>{label}</dt><dd title={String(value)}>{typeof value === 'number' ? numero(value) : value}</dd></div>)}</dl>
}
function TablaPaginada<T>({ titulo, filas, columnas }: { titulo: string; filas: T[]; columnas: { nombre: string; celda: (fila: T) => ReactNode }[] }) {
  const [pagina, setPagina] = useState(0)
  const paginas = Math.max(1, Math.ceil(filas.length / PAGINA))
  const actual = Math.min(pagina, paginas - 1)
  return <div className="global-profile-table"><div className="global-profile-scroll" tabIndex={0} role="region" aria-label={titulo}>
    <table><caption>{titulo}</caption><thead><tr>{columnas.map(c => <th scope="col" key={c.nombre}>{c.nombre}</th>)}</tr></thead>
      <tbody>{filas.slice(actual * PAGINA, (actual + 1) * PAGINA).map((f, j) => <tr key={actual * PAGINA + j}>{columnas.map(c => <td key={c.nombre}>{c.celda(f)}</td>)}</tr>)}</tbody></table></div>
    {filas.length === 0 ? <p>No hay registros en esta lista.</p> : <div className="global-profile-pagination">
      <button type="button" disabled={actual === 0} onClick={() => setPagina(actual - 1)} aria-label={`Página anterior: ${titulo}`}>Anterior</button>
      <span role="status">Página {numero(actual + 1)} de {numero(paginas)} · {numero(filas.length)} registros</span>
      <button type="button" disabled={actual + 1 === paginas} onClick={() => setPagina(actual + 1)} aria-label={`Página siguiente: ${titulo}`}>Siguiente</button>
    </div>}</div>
}
function mensajeHTTP(status: number, body: unknown): string {
  if (status === 404) return 'El conjunto activo no tiene un perfil global. Los conjuntos anteriores a esta función pueden carecer de reporte; no se reconstruye desde las diez series.'
  if (status === 409) return objeto(body) && body.detail === 'No existe un conjunto de datos activo.'
    ? 'No existe un conjunto activo. Actualice la consulta de series.'
    : 'El perfil almacenado es inconsistente, supera el límite permitido o utiliza una versión no compatible.'
  if (status === 503) return 'La base de datos o el almacenamiento del panorama global no está disponible.'
  if (status === 500) return 'El servidor no pudo completar la consulta. Inténtelo nuevamente.'
  return 'No se pudo recuperar el panorama global. Actualice las series o vuelva a consultar.'
}

export default function PerfilGlobalDatos({ conjuntoIdActivo, estadoConjunto, activo, actualizarSeries }: Props) {
  const id = useId()
  const [estado, setEstado] = useState<Estado>({ kind: 'idle' })
  const solicitud = useRef<AbortController | null>(null)
  const version = useRef(0)
  const contexto = useRef({ conjuntoIdActivo, estadoConjunto, activo })
  useLayoutEffect(() => {
    function invalidar() { version.current++; solicitud.current?.abort(); solicitud.current = null }
    contexto.current = { conjuntoIdActivo, estadoConjunto, activo }
    invalidar()
    return invalidar
  }, [conjuntoIdActivo, estadoConjunto, activo])
  async function consultar() {
    if (!activo || conjuntoIdActivo === null || estadoConjunto !== 'available' || solicitud.current) return
    const esperado = conjuntoIdActivo, revision = ++version.current
    const controller = new AbortController(); solicitud.current = controller
    const vigente = () => !controller.signal.aborted && version.current === revision && contexto.current.activo &&
      contexto.current.conjuntoIdActivo === esperado && contexto.current.estadoConjunto === 'available'
    setEstado({ kind: 'loading', signal: controller.signal })
    try {
      let response: Response
      try { response = await fetch('/api/datasets/active/perfil-global', { method: 'GET', signal: controller.signal, cache: 'no-store' }) }
      catch { if (vigente()) setEstado({ kind: 'error', titulo: 'Error de conexión', mensaje: 'No se pudo conectar para consultar el panorama global. Compruebe la conexión y reintente.' }); return }
      if (!vigente()) return
      let body: unknown
      try { body = await response.json() } catch {
        if (vigente()) setEstado({ kind: 'error', titulo: response.ok ? 'JSON inválido' : `Error HTTP ${response.status}`,
          mensaje: response.ok ? 'La respuesta no contiene JSON válido. No se muestran estadísticas.' : mensajeHTTP(response.status, null) })
        return
      }
      if (!vigente()) return
      if (!response.ok) setEstado({ kind: 'error', titulo: `Error HTTP ${response.status}`, mensaje: mensajeHTTP(response.status, body), actualizar: response.status === 409 })
      else if (!esPerfilGlobal(body)) setEstado({ kind: 'error', titulo: 'Respuesta incompatible', mensaje: 'El reporte tiene una estructura o agregados inconsistentes. No se muestran estadísticas.', actualizar: true })
      else if (body.conjunto_id !== esperado) setEstado({ kind: 'error', titulo: 'Respuesta incompatible', mensaje: 'El conjunto activo cambió. Actualice la consulta de series antes de consultar nuevamente.', actualizar: true })
      else setEstado({ kind: 'success', datos: body })
    } finally { if (solicitud.current === controller) solicitud.current = null }
  }
  const confirmado = conjuntoIdActivo !== null && estadoConjunto === 'available'
  const datos = confirmado && activo && estado.kind === 'success' && estado.datos.conjunto_id === conjuntoIdActivo ? estado.datos : null
  const cargando = estado.kind === 'loading' && !estado.signal.aborted && activo
  const r = datos?.reporte
  return <section className="global-profile-card" aria-labelledby={`${id}-titulo`}>
    <h2 id={`${id}-titulo`}>Panorama descriptivo global</h2>
    <p>Descripción del archivo CSV completo asociado al conjunto activo. La consulta no genera pronósticos.</p>
    {!confirmado && <p role="status" className="global-profile-note">{estadoConjunto === 'loading' ? 'Esperando la confirmación de las series activas…' : estadoConjunto === 'no-active' ? 'No hay un conjunto activo confirmado. Cargue un CSV para comenzar.' : 'No se ha podido confirmar un conjunto con series disponibles. Actualice la consulta de series.'}</p>}
    <div className="global-profile-actions">
      <button type="button" disabled={!confirmado || !activo || cargando} onClick={() => { void consultar() }}>Consultar panorama global</button>
      {(!confirmado || (estado.kind === 'error' && estado.actualizar)) && <button type="button" disabled={!activo || estadoConjunto === 'loading'} onClick={actualizarSeries}>Actualizar consulta de series</button>}
    </div>
    {confirmado && estado.kind === 'idle' && <p className="global-profile-note">Listo para consultar el conjunto {conjuntoIdActivo}. No se solicita el panorama automáticamente.</p>}
    {cargando && <p role="status" className="global-profile-note">Consultando panorama global…</p>}
    {confirmado && estado.kind === 'loading' && !cargando && activo && <p role="status" className="global-profile-note">Consulta cancelada. Puede solicitar el panorama nuevamente.</p>}
    {confirmado && activo && estado.kind === 'error' && <div role="alert" className="global-profile-error"><strong>{estado.titulo}</strong><p>{estado.mensaje}</p><p>Puede reintentar con «Consultar panorama global».</p></div>}
    {datos && r && <div key={`${datos.conjunto_id}:${datos.sha256_csv}`}>
      <h3>Identidad y procedencia</h3>
      <Indicadores items={[["Archivo CSV", r.fuente.nombre_archivo], ["Conjunto activo", datos.conjunto_id], ["Versión del perfil", datos.version_perfil]]} />
      <label className="global-profile-hash">SHA-256 del archivo<input readOnly aria-label="SHA-256 del archivo" value={datos.sha256_csv} onFocus={e => e.currentTarget.select()} /></label>
      <p className="global-profile-notice">Este panorama describe retrospectivamente el archivo completo. No se utiliza para seleccionar modelos, variables o hiperparámetros con información posterior a las fechas de corte.</p>
      <p className="global-profile-note">{r.interpretacion_temporal.nota}</p>
      <div className="global-profile-comparison"><section><h3>Archivo completo</h3><Indicadores items={[["Filas globales examinadas", r.volumen.filas_examinadas], ["Filas válidas", r.volumen.filas_validas], ["Filas inválidas", r.volumen.filas_invalidas], ["Tiendas distintas", r.volumen.tiendas_distintas], ["Familias distintas", r.volumen.familias_distintas], ["Combinaciones observadas", r.volumen.combinaciones_observadas]]} /></section>
        <section><h3>Subconjunto de trabajo</h3><Indicadores items={[["Series importadas a SQLite", datos.series_importadas], ["Observaciones importadas a SQLite", datos.observaciones_importadas]]} /><p>La aplicación conserva las series habilitadas. Las filas globales no representan todas observaciones importadas ni utilizadas para entrenar.</p></section></div>
      <details className="global-profile-section" open><summary>Cobertura temporal y fechas sin registro</summary>
        <Indicadores items={[["Primera fecha observada", fecha(r.temporalidad.primera_fecha)], ["Última fecha observada", fecha(r.temporalidad.ultima_fecha)], ["Fechas distintas", r.temporalidad.fechas_distintas], ["Días naturales del intervalo", r.temporalidad.dias_calendario_inclusivos], ["Fechas sin registro global", r.temporalidad.fechas_sin_registro_global.length]]} />
        <p>Una fecha sin registro no es una venta cero. Las ausencias globales no describen automáticamente las ausencias de cada serie; no se completan registros.</p>
        <details><summary>Consultar fechas ausentes</summary><TablaPaginada titulo="Fechas sin ningún registro global" filas={r.temporalidad.fechas_sin_registro_global} columnas={[{ nombre: 'Fecha ausente', celda: f => <time dateTime={f}>{fecha(f)}</time> }]} /></details>
        <details><summary>Frecuencias por fecha y año</summary><TablaPaginada titulo="Filas observadas por fecha" filas={r.temporalidad.filas_por_fecha} columnas={[{ nombre: 'Fecha', celda: f => <time dateTime={f.fecha}>{fecha(f.fecha)}</time> }, { nombre: 'Filas', celda: f => numero(f.filas) }]} />
          <TablaPaginada titulo="Filas observadas por año" filas={r.temporalidad.filas_por_anio} columnas={[{ nombre: 'Año', celda: a => String(a.anio) }, { nombre: 'Filas', celda: a => numero(a.filas) }]} /></details>
      </details>
      <details className="global-profile-section" open><summary>Ventas del archivo completo</summary><Indicadores items={[["Valor acumulado de sales", r.ventas.suma], ["Promedio por registro", r.ventas.promedio_por_registro], ["Mínimo de sales", r.ventas.minimo], ["Máximo de sales", r.ventas.maximo], ["Registros con ventas cero", r.ventas.registros_cero], ["Porcentaje de ceros (%)", r.ventas.porcentaje_cero], ["Registros fraccionarios", r.ventas.registros_fraccionarios], ["Porcentaje de fraccionarios (%)", r.ventas.porcentaje_fraccionarios], ["Registros con ventas positivas", r.ventas.registros_positivos]]} /><p>No se atribuyen a sales unidades monetarias ni físicas adicionales.</p></details>
      <details className="global-profile-section"><summary>Promociones descriptivas</summary><Indicadores items={[["Registros con promoción", r.promociones.registros_positivos], ["Registros sin promoción", r.promociones.registros_cero], ["Porcentaje con promoción (%)", r.promociones.porcentaje_con_promocion], ["Suma literal de onpromotion", r.promociones.suma_conteos_onpromotion], ["Mínimo de onpromotion", r.promociones.minimo], ["Máximo de onpromotion", r.promociones.maximo], ["Promedio de onpromotion", r.promociones.promedio_por_registro]]} /><p>onpromotion es un conteo registrado. No representa descuentos monetarios ni artículos únicos y no demuestra causalidad sobre las ventas.</p></details>
      <details className="global-profile-section"><summary>Distribuciones globales</summary>
        <TablaPaginada titulo="Registros por tienda" filas={r.distribuciones.por_tienda} columnas={[{ nombre: 'Tienda', celda: t => numero(t.store_nbr) }, { nombre: 'Filas', celda: t => numero(t.filas) }]} />
        <TablaPaginada titulo="Registros y sales por familia" filas={r.distribuciones.por_familia} columnas={[{ nombre: 'Familia', celda: f => f.family }, { nombre: 'Filas', celda: f => numero(f.filas) }, { nombre: 'Valor acumulado de sales', celda: f => numero(f.ventas) }]} />
        <p>Las combinaciones de mayor volumen del CSV no equivalen a las diez series oficiales de modelado.</p><p>{r.distribuciones.criterio_mayores}</p>
        <TablaPaginada titulo="Combinaciones de mayor volumen del archivo" filas={r.distribuciones.combinaciones_mayor_volumen} columnas={[{ nombre: 'Tienda', celda: c => numero(c.store_nbr) }, { nombre: 'Familia', celda: c => c.family }, { nombre: 'Filas', celda: c => numero(c.filas) }, { nombre: 'Valor acumulado de sales', celda: c => numero(c.ventas) }]} />
      </details>
      <section className="global-profile-section"><h3>Advertencias del panorama global</h3><ul>{r.advertencias.map((a, j) => <li key={j}>{a}</li>)}</ul><p>Estas estadísticas descriptivas no demuestran significación estadística ni causalidad.</p></section>
    </div>}
  </section>
}
