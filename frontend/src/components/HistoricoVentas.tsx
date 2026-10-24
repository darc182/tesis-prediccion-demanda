import { useEffect, useRef, useState, type FormEvent } from 'react'
import type { SerieTemporal } from './SelectorSerie'

type Seleccion = { conjuntoId: number; serie: SerieTemporal }
type Observacion = { fecha: string; sales: number; onpromotion: number }
type RespuestaHistorico = {
  conjunto_id: number
  serie: SerieTemporal
  observaciones: Observacion[]
}
type Periodo = { desde: string; hasta: string }
type EstadoHistorico =
  | { kind: 'idle' | 'loading' }
  | { kind: 'date-error' | 'network-error' | 'http-error' | 'invalid-response'; message: string }
  | { kind: 'success'; data: RespuestaHistorico; periodo: Periodo }

type Props = { seleccion: Seleccion | null }

const TAMANO_PAGINA = 25
const MILISEGUNDOS_DIA = 24 * 60 * 60 * 1000
const formatoVentas = new Intl.NumberFormat('es-EC', { maximumFractionDigits: 6 })
const formatoEje = new Intl.NumberFormat('es-EC', { maximumSignificantDigits: 4 })

function instanteFecha(fecha: string): number | null {
  if (!/^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(fecha)) return null
  const instante = Date.parse(`${fecha}T00:00:00Z`)
  if (!Number.isFinite(instante) || new Date(instante).toISOString().slice(0, 10) !== fecha) return null
  return instante
}

function esRespuestaHistorico(
  value: unknown, seleccion: Seleccion, periodo: Periodo,
): value is RespuestaHistorico {
  if (typeof value !== 'object' || value === null) return false
  const data = value as Record<string, unknown>
  if (data.conjunto_id !== seleccion.conjuntoId ||
      typeof data.serie !== 'object' || data.serie === null ||
      !Array.isArray(data.observaciones)) return false

  const serie = data.serie as Record<string, unknown>
  if (serie.id !== seleccion.serie.id ||
      serie.store_nbr !== seleccion.serie.store_nbr ||
      serie.family !== seleccion.serie.family) return false

  let fechaAnterior = -Infinity
  return data.observaciones.every((item: unknown) => {
    if (typeof item !== 'object' || item === null) return false
    const observacion = item as Record<string, unknown>
    if (typeof observacion.fecha !== 'string') return false
    const instante = instanteFecha(observacion.fecha)
    if (instante === null || instante <= fechaAnterior ||
        (periodo.desde && observacion.fecha < periodo.desde) ||
        (periodo.hasta && observacion.fecha > periodo.hasta) ||
        typeof observacion.sales !== 'number' || !Number.isFinite(observacion.sales) ||
        typeof observacion.onpromotion !== 'number' ||
        !Number.isInteger(observacion.onpromotion) || observacion.onpromotion < 0) return false
    fechaAnterior = instante
    return true
  })
}

function descripcionPeriodo({ desde, hasta }: Periodo): string {
  if (desde && hasta) return `Del ${desde} al ${hasta}`
  if (desde) return `Desde ${desde}`
  if (hasta) return `Hasta ${hasta}`
  return 'Todo el histórico disponible'
}

function GraficoVentas({ observaciones }: { observaciones: Observacion[] }) {
  const ancho = 760
  const alto = 320
  const izquierda = 76
  const derecha = 48
  const arriba = 24
  const abajo = 60
  const anchoGrafico = ancho - izquierda - derecha
  const altoGrafico = alto - arriba - abajo
  const inicio = instanteFecha(observaciones[0].fecha) as number
  const fin = instanteFecha(observaciones[observaciones.length - 1].fecha) as number
  const minimo = observaciones.reduce((valor, fila) => Math.min(valor, fila.sales), 0)
  const maximoReal = observaciones.reduce((valor, fila) => Math.max(valor, fila.sales), 0)
  const maximo = maximoReal === minimo ? minimo + 1 : maximoReal
  const posicionX = (fecha: string) => {
    const instante = instanteFecha(fecha) as number
    return inicio === fin
      ? izquierda + anchoGrafico / 2
      : izquierda + (instante - inicio) / (fin - inicio) * anchoGrafico
  }
  const posicionY = (sales: number) => arriba + (maximo - sales) / (maximo - minimo) * altoGrafico

  // Cada tramo contiene solo fechas consecutivas; una fecha ausente corta la línea.
  const tramos: Observacion[][] = []
  for (const fila of observaciones) {
    const tramo = tramos[tramos.length - 1]
    const ultima = tramo?.[tramo.length - 1]
    if (!ultima || (instanteFecha(fila.fecha) as number) - (instanteFecha(ultima.fecha) as number) !== MILISEGUNDOS_DIA) {
      tramos.push([fila])
    } else {
      tramo.push(fila)
    }
  }

  const marcasY = Array.from({ length: 5 }, (_, indice) => minimo + (maximo - minimo) * indice / 4)
  const marcasX = inicio === fin ? [inicio] : Array.from(
    { length: 5 }, (_, indice) => inicio + (fin - inicio) * indice / 4,
  ).filter((instante, indice, marcas) => indice === 0 ||
    new Date(instante).toISOString().slice(0, 10) !== new Date(marcas[indice - 1]).toISOString().slice(0, 10))

  return (
    <figure className="history-chart">
      <svg viewBox={`0 0 ${ancho} ${alto}`} role="img" aria-labelledby="history-chart-title history-chart-desc">
        <title id="history-chart-title">Ventas diarias de la serie seleccionada</title>
        <desc id="history-chart-desc">
          Eje horizontal por fecha real y eje vertical por ventas. Las líneas se interrumpen cuando faltan días; los ceros registrados se muestran en el eje.
        </desc>
        {marcasY.map((valor, indice) => (
          <g key={indice}>
            <line x1={izquierda} x2={ancho - derecha} y1={posicionY(valor)} y2={posicionY(valor)} className="chart-grid" />
            <text x={izquierda - 10} y={posicionY(valor) + 4} textAnchor="end" className="chart-label">
              {formatoEje.format(valor)}
            </text>
          </g>
        ))}
        <line x1={izquierda} x2={izquierda} y1={arriba} y2={alto - abajo} className="chart-axis" />
        <line x1={izquierda} x2={ancho - derecha} y1={alto - abajo} y2={alto - abajo} className="chart-axis" />
        {marcasX.map((instante) => {
          const x = inicio === fin ? izquierda + anchoGrafico / 2 : izquierda + (instante - inicio) / (fin - inicio) * anchoGrafico
          return (
            <text key={instante} x={x} y={alto - abajo + 22} textAnchor="middle" className="chart-label">
              {new Date(instante).toISOString().slice(0, 10)}
            </text>
          )
        })}
        <text x={izquierda + anchoGrafico / 2} y={alto - 7} textAnchor="middle" className="chart-axis-title">Fecha</text>
        <text transform={`translate(18 ${arriba + altoGrafico / 2}) rotate(-90)`} textAnchor="middle" className="chart-axis-title">Ventas</text>
        {tramos.map((tramo, indice) => (
          <g key={indice}>
            {tramo.length > 1 && (
              <polyline
                points={tramo.map((fila) => `${posicionX(fila.fecha)},${posicionY(fila.sales)}`).join(' ')}
                className="chart-line"
              />
            )}
            {(observaciones.length <= 60 || tramo.length === 1) && tramo.map((fila) => (
              <circle key={fila.fecha} cx={posicionX(fila.fecha)} cy={posicionY(fila.sales)} r="3.5" className="chart-point">
                <title>{fila.fecha}: {formatoVentas.format(fila.sales)} ventas</title>
              </circle>
            ))}
          </g>
        ))}
      </svg>
      <figcaption>Los días sin registro interrumpen la línea; no se interpretan como ventas cero.</figcaption>
    </figure>
  )
}

function HistoricoVentas({ seleccion }: Props) {
  const [desde, setDesde] = useState('')
  const [hasta, setHasta] = useState('')
  const [estado, setEstado] = useState<EstadoHistorico>({ kind: 'idle' })
  const [pagina, setPagina] = useState(1)
  const solicitud = useRef<AbortController | null>(null)

  useEffect(() => () => solicitud.current?.abort(), [])

  async function consultar(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (seleccion === null) return
    solicitud.current?.abort()
    solicitud.current = null
    if (desde && hasta && desde > hasta) {
      setEstado({ kind: 'date-error', message: 'Desde no puede ser posterior a Hasta.' })
      return
    }

    const controller = new AbortController()
    solicitud.current = controller
    const periodo = { desde, hasta }
    const parametros = new URLSearchParams()
    if (desde) parametros.set('desde', desde)
    if (hasta) parametros.set('hasta', hasta)
    const query = parametros.toString()
    const url = `/api/datasets/active/series/${seleccion.serie.id}/sales${query ? `?${query}` : ''}`
    setPagina(1)
    setEstado({ kind: 'loading' })

    try {
      let response: Response
      try {
        response = await fetch(url, { signal: controller.signal })
      } catch {
        if (!controller.signal.aborted) setEstado({
          kind: 'network-error', message: 'No se pudo conectar con FastAPI para consultar el histórico.',
        })
        return
      }

      if (controller.signal.aborted) return
      if (!response.ok) {
        setEstado({ kind: 'http-error', message: `La consulta del histórico respondió con HTTP ${response.status}.` })
        return
      }

      let body: unknown
      try {
        body = await response.json()
      } catch {
        if (!controller.signal.aborted) setEstado({ kind: 'invalid-response', message: 'La API no devolvió un JSON válido.' })
        return
      }

      if (controller.signal.aborted) return
      if (!esRespuestaHistorico(body, seleccion, periodo)) {
        setEstado({ kind: 'invalid-response', message: 'La respuesta no corresponde a la serie activa o contiene datos inválidos.' })
      } else {
        setEstado({ kind: 'success', data: body, periodo })
      }
    } finally {
      if (solicitud.current === controller) solicitud.current = null
    }
  }

  const observaciones = estado.kind === 'success' ? estado.data.observaciones : []
  const paginas = Math.ceil(observaciones.length / TAMANO_PAGINA)
  const filasPagina = observaciones.slice((pagina - 1) * TAMANO_PAGINA, pagina * TAMANO_PAGINA)

  return (
    <section className="history-card" aria-labelledby="history-heading">
      <h2 id="history-heading">Ventas históricas</h2>
      {seleccion === null ? (
        <p role="status">Seleccione una tienda y una familia válidas para consultar su histórico.</p>
      ) : (
        <>
          <p className="history-series">Tienda <strong>{seleccion.serie.store_nbr}</strong> · Familia <strong>{seleccion.serie.family}</strong></p>
          <form className="history-form" onSubmit={(event) => { void consultar(event) }} aria-busy={estado.kind === 'loading'}>
            <div>
              <label htmlFor="history-from">Desde</label>
              <input id="history-from" type="date" value={desde} onChange={(event) => setDesde(event.target.value)} />
            </div>
            <div>
              <label htmlFor="history-to">Hasta</label>
              <input id="history-to" type="date" value={hasta} onChange={(event) => setHasta(event.target.value)} />
            </div>
            <button type="submit" disabled={estado.kind === 'loading'}>Consultar histórico</button>
          </form>

          {estado.kind === 'idle' && <p className="history-message">Indique fechas si desea limitar el periodo y pulse «Consultar histórico».</p>}
          {estado.kind === 'loading' && <p className="history-message" role="status">Cargando histórico…</p>}
          {(estado.kind === 'date-error' || estado.kind === 'network-error' ||
            estado.kind === 'http-error' || estado.kind === 'invalid-response') && (
            <p className="history-message history-error" role="alert">{estado.message}</p>
          )}
          {estado.kind === 'success' && (
            <div className="history-results">
              <p className="history-period"><strong>Periodo consultado:</strong> {descripcionPeriodo(estado.periodo)}</p>
              {observaciones.length === 0 ? (
                <p role="status">No hay observaciones registradas en este periodo.</p>
              ) : (
                <>
                  <p>{observaciones.length.toLocaleString('es-EC')} observaciones registradas.</p>
                  <GraficoVentas observaciones={observaciones} />
                  <div className="history-table-wrap">
                    <table>
                      <caption>Observaciones históricas de la serie seleccionada</caption>
                      <thead><tr><th scope="col">Fecha</th><th scope="col">Ventas (sales)</th><th scope="col">Promoción (onpromotion)</th></tr></thead>
                      <tbody>
                        {filasPagina.map((fila) => (
                          <tr key={fila.fecha}>
                            <td>{fila.fecha}</td>
                            <td>{formatoVentas.format(fila.sales)}</td>
                            <td>{fila.onpromotion}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <nav className="history-pagination" aria-label="Páginas de observaciones">
                    <button type="button" disabled={pagina === 1} onClick={() => setPagina(pagina - 1)}>Anterior</button>
                    <span>Página {pagina} de {paginas}</span>
                    <button type="button" disabled={pagina === paginas} onClick={() => setPagina(pagina + 1)}>Siguiente</button>
                  </nav>
                </>
              )}
            </div>
          )}
        </>
      )}
    </section>
  )
}

export default HistoricoVentas
