import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import HistoricoVentas from './components/HistoricoVentas'
import SelectorSerie, { type SerieTemporal, type SeriesState } from './components/SelectorSerie'
import './App.css'

type ConnectionStatus = 'connecting' | 'connected' | 'error'
type Vista = 'datos' | 'historico' | 'pronosticos' | 'comparacion' | 'simulacion'

const apartados: { id: Vista; nombre: string }[] = [
  { id: 'datos', nombre: 'Datos' },
  { id: 'historico', nombre: 'Histórico' },
  { id: 'pronosticos', nombre: 'Pronósticos' },
  { id: 'comparacion', nombre: 'Comparación' },
  { id: 'simulacion', nombre: 'Simulación' },
]

type UploadResult = {
  exito: boolean
  conjunto_id: number | null
  sha256: string | null
  series_importadas: number
  observaciones_importadas: number
  errores: string[]
  advertencias: string[]
}

type UploadState =
  | { kind: 'idle' | 'processing' }
  | { kind: 'client-error' | 'network-error' | 'response-error'; message: string }
  | { kind: 'server-error'; result: UploadResult; status: number }
  | { kind: 'success'; result: UploadResult }

type SeriesResponse = {
  conjunto_id: number | null
  series: SerieTemporal[]
}

function isSeriesResponse(value: unknown): value is SeriesResponse {
  if (typeof value !== 'object' || value === null) return false
  const data = value as Record<string, unknown>
  const validId = data.conjunto_id === null ||
    (typeof data.conjunto_id === 'number' && Number.isInteger(data.conjunto_id) && data.conjunto_id > 0)
  if (!validId || !Array.isArray(data.series)) return false
  if (data.conjunto_id === null && data.series.length > 0) return false

  const keys = new Set<string>()
  return data.series.every((item: unknown) => {
    if (typeof item !== 'object' || item === null) return false
    const serie = item as Record<string, unknown>
    if (typeof serie.id !== 'number' || !Number.isInteger(serie.id) || serie.id <= 0 ||
        typeof serie.store_nbr !== 'number' || !Number.isInteger(serie.store_nbr) || serie.store_nbr <= 0 ||
        typeof serie.family !== 'string' || !serie.family.trim()) return false
    const key = JSON.stringify([serie.store_nbr, serie.family])
    if (keys.has(key)) return false
    keys.add(key)
    return true
  })
}

function isUploadResult(value: unknown): value is UploadResult {
  if (typeof value !== 'object' || value === null) return false
  const data = value as Record<string, unknown>

  return (
    typeof data.exito === 'boolean' &&
    (data.conjunto_id === null ||
      (typeof data.conjunto_id === 'number' && Number.isInteger(data.conjunto_id))) &&
    (data.sha256 === null || typeof data.sha256 === 'string') &&
    typeof data.series_importadas === 'number' &&
    Number.isInteger(data.series_importadas) &&
    data.series_importadas >= 0 &&
    typeof data.observaciones_importadas === 'number' &&
    Number.isInteger(data.observaciones_importadas) &&
    data.observaciones_importadas >= 0 &&
    Array.isArray(data.errores) &&
    data.errores.every((error) => typeof error === 'string') &&
    Array.isArray(data.advertencias) &&
    data.advertencias.every((advertencia) => typeof advertencia === 'string')
  )
}

function formatoTamano(bytes: number): string {
  if (bytes < 1024) return `${bytes.toLocaleString('es-EC')} B`
  const unidad = bytes < 1024 * 1024 ? 'KB' : 'MB'
  const divisor = unidad === 'KB' ? 1024 : 1024 * 1024
  return `${(bytes / divisor).toLocaleString('es-EC', { maximumFractionDigits: 1 })} ${unidad}`
}

function App() {
  const [vista, setVista] = useState<Vista>('datos')
  const [connection, setConnection] = useState<ConnectionStatus>('connecting')
  const [file, setFile] = useState<File | null>(null)
  const [upload, setUpload] = useState<UploadState>({ kind: 'idle' })
  const [seriesState, setSeriesState] = useState<SeriesState>({ kind: 'loading' })
  const [selectedStore, setSelectedStore] = useState<number | null>(null)
  const [selectedFamily, setSelectedFamily] = useState<string | null>(null)
  const seriesRequest = useRef<AbortController | null>(null)

  const refreshSeries = useCallback(async () => {
    seriesRequest.current?.abort()
    const controller = new AbortController()
    seriesRequest.current = controller

    try {
      let response: Response
      try {
        response = await fetch('/api/datasets/active/series', { signal: controller.signal })
      } catch {
        if (!controller.signal.aborted) setSeriesState({
          kind: 'network-error',
          message: 'No se pudo conectar para consultar las series. Compruebe la conexión con FastAPI.',
        })
        return
      }

      if (controller.signal.aborted) return
      if (!response.ok) {
        setSeriesState({
          kind: 'response-error',
          message: `No se pudieron consultar las series (HTTP ${response.status}).`,
        })
        return
      }

      let body: unknown
      try {
        body = await response.json()
      } catch {
        if (!controller.signal.aborted) setSeriesState({
          kind: 'response-error',
          message: 'La API no devolvió un JSON válido para las series.',
        })
        return
      }

      if (controller.signal.aborted) return
      if (!isSeriesResponse(body)) {
        setSeriesState({
          kind: 'response-error',
          message: 'La respuesta de series de la API tiene un formato inválido.',
        })
      } else if (body.conjunto_id === null) {
        setSeriesState({ kind: 'no-active' })
      } else if (body.series.length === 0) {
        setSeriesState({ kind: 'empty' })
      } else {
        setSeriesState({ kind: 'available', conjuntoId: body.conjunto_id, series: body.series })
      }
    } finally {
      if (seriesRequest.current === controller) seriesRequest.current = null
    }
  }, [])

  useEffect(() => {
    void refreshSeries()
    return () => seriesRequest.current?.abort()
  }, [refreshSeries])

  useEffect(() => {
    const controller = new AbortController()

    async function checkConnection() {
      try {
        const response = await fetch('/health', { signal: controller.signal })
        if (!response.ok) throw new Error('La solicitud falló')

        const data: unknown = await response.json()
        if (
          typeof data !== 'object' ||
          data === null ||
          !('status' in data) ||
          data.status !== 'ok'
        ) {
          throw new Error('Respuesta de salud inesperada')
        }

        setConnection('connected')
      } catch {
        if (!controller.signal.aborted) setConnection('error')
      }
    }

    void checkConnection()
    return () => controller.abort()
  }, [])

  const connectionMessages: Record<ConnectionStatus, string> = {
    connecting: 'Comprobando conexión con FastAPI…',
    connected: 'FastAPI conectado',
    error: 'No se pudo conectar con FastAPI',
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (upload.kind === 'processing') return
    if (!file || !file.name.toLowerCase().endsWith('.csv')) {
      setUpload({ kind: 'client-error', message: 'Seleccione un archivo con extensión .csv.' })
      return
    }

    const formData = new FormData()
    formData.append('file', file)
    setUpload({ kind: 'processing' })

    let response: Response
    try {
      response = await fetch('/api/datasets/upload', {
        method: 'POST',
        body: formData,
      })
    } catch {
      setUpload({
        kind: 'network-error',
        message: 'No se pudo enviar el archivo. Compruebe la conexión con FastAPI e inténtelo de nuevo.',
      })
      setConnection('error')
      return
    }

    let body: unknown
    try {
      body = await response.json()
    } catch {
      setUpload({
        kind: 'response-error',
        message: `El servidor respondió con HTTP ${response.status}, pero no devolvió un JSON válido.`,
      })
      return
    }

    if (!isUploadResult(body)) {
      setUpload({
        kind: 'response-error',
        message: `El servidor respondió con HTTP ${response.status}, pero el JSON no tiene el formato esperado.`,
      })
      return
    }

    if (!response.ok || !body.exito) {
      setUpload({ kind: 'server-error', result: body, status: response.status })
      return
    }

    if (body.conjunto_id === null) {
      setUpload({
        kind: 'response-error',
        message: 'La respuesta de carga no incluye el identificador del conjunto.',
      })
      return
    }

    setUpload({ kind: 'success', result: body })
    setSelectedStore(null)
    setSelectedFamily(null)
    setSeriesState({ kind: 'loading' })
    void refreshSeries()
  }

  function handleRetrySeries() {
    setSelectedStore(null)
    setSelectedFamily(null)
    setSeriesState({ kind: 'loading' })
    void refreshSeries()
  }

  function handleStoreChange(store: number | null) {
    if (store !== null &&
        (seriesState.kind !== 'available' || !seriesState.series.some((serie) => serie.store_nbr === store))) return
    setSelectedStore(store)
    setSelectedFamily((family) =>
      family !== null && seriesState.kind === 'available' &&
      seriesState.series.some((serie) => serie.store_nbr === store && serie.family === family)
        ? family : null,
    )
  }

  function handleFamilyChange(family: string | null) {
    if (seriesState.kind !== 'available' || selectedStore === null) return
    setSelectedFamily(
      family !== null && seriesState.series.some(
        (serie) => serie.store_nbr === selectedStore && serie.family === family,
      ) ? family : null,
    )
  }

  const processing = upload.kind === 'processing'
  const serieSeleccionada = seriesState.kind === 'available' &&
    selectedStore !== null && selectedFamily !== null
    ? seriesState.series.find((serie) =>
      serie.store_nbr === selectedStore && serie.family === selectedFamily)
    : undefined
  const seleccionHistorico = seriesState.kind === 'available' && serieSeleccionada
    ? { conjuntoId: seriesState.conjuntoId, serie: serieSeleccionada }
    : null

  return (
    <main className="app-page">
      <header className="page-header">
        <p className="eyebrow">Análisis de inventario</p>
        <h1>Predicción de demanda</h1>
        <p className="intro">
          Gestione los datos y consulte las ventas históricas por tienda y familia de producto.
        </p>
        <p className={`connection-status connection-status--${connection}`} role="status" aria-live="polite">
          {connectionMessages[connection]}
        </p>
      </header>

      <nav className="main-nav" aria-label="Navegación principal">
        {apartados.map((apartado) => (
          <button
            key={apartado.id}
            type="button"
            aria-current={vista === apartado.id ? 'page' : undefined}
            onClick={() => setVista(apartado.id)}
          >
            {apartado.nombre}
          </button>
        ))}
      </nav>

      <section className="app-view" aria-label="Datos" hidden={vista !== 'datos'}>
        <section className="upload-card" aria-labelledby="upload-heading">
          <h2 id="upload-heading">Seleccionar archivo</h2>
          <form onSubmit={(event) => { void handleSubmit(event) }} aria-busy={processing}>
            <label htmlFor="csv-file">Archivo CSV</label>
            <input
              id="csv-file"
              type="file"
              accept=".csv,text/csv"
              disabled={processing}
              aria-describedby="file-help"
              onChange={(event) => {
                setFile(event.target.files?.[0] ?? null)
                setUpload({ kind: 'idle' })
              }}
            />
            <p id="file-help" className="help-text">
              Se comprobarán el formato y las series disponibles al enviar el archivo.
            </p>
            {file && (
              <p className="file-details">
                Seleccionado: <strong>{file.name}</strong> · {formatoTamano(file.size)}
              </p>
            )}
            <button type="submit" disabled={processing}>Cargar y validar datos</button>
          </form>
        </section>

        {upload.kind === 'processing' && (
          <section className="feedback feedback--processing" role="status" aria-live="polite">
            <h2>Procesando archivo…</h2>
            <p>La validación y la carga de archivos grandes pueden tardar varios minutos.</p>
          </section>
        )}

        {(upload.kind === 'client-error' || upload.kind === 'network-error' ||
          upload.kind === 'response-error') && (
          <section className="feedback feedback--error" role="alert">
            <h2>No se pudo completar la carga</h2>
            <p>{upload.message}</p>
          </section>
        )}

        {upload.kind === 'server-error' && (
          <section className="feedback feedback--error" role="alert">
            <h2>Carga rechazada</h2>
            <p>Respuesta HTTP {upload.status}.</p>
            {upload.result.errores.length > 0 ? (
              <ul>{upload.result.errores.map((error, index) => <li key={index}>{error}</li>)}</ul>
            ) : (
              <p>El servidor no proporcionó un mensaje de error.</p>
            )}
          </section>
        )}

        {upload.kind === 'success' && (
          <section className="feedback feedback--success" role="status" aria-live="polite">
            <h2>Datos cargados correctamente</h2>
            <dl className="result-summary">
              <div><dt>Conjunto</dt><dd>{upload.result.conjunto_id}</dd></div>
              <div><dt>Series importadas</dt><dd>{upload.result.series_importadas}</dd></div>
              <div><dt>Observaciones importadas</dt><dd>{upload.result.observaciones_importadas}</dd></div>
            </dl>
            {upload.result.advertencias.length > 0 && (
              <div className="warnings">
                <h3>Advertencias</h3>
                <ul>{upload.result.advertencias.map((warning, index) => <li key={index}>{warning}</li>)}</ul>
              </div>
            )}
          </section>
        )}
      </section>

      <section className="app-view" aria-label="Histórico" hidden={vista !== 'historico'}>
        <SelectorSerie
          state={seriesState}
          selectedStore={selectedStore}
          selectedFamily={selectedFamily}
          onStoreChange={handleStoreChange}
          onFamilyChange={handleFamilyChange}
          onRetry={handleRetrySeries}
        />
        <HistoricoVentas
          key={seleccionHistorico ? `${seleccionHistorico.conjuntoId}:${seleccionHistorico.serie.id}` : 'sin-serie'}
          seleccion={seleccionHistorico}
          activo={vista === 'historico'}
        />
      </section>

      {vista !== 'datos' && vista !== 'historico' && (
        <section className="pending-view" aria-labelledby="pending-heading">
          <h2 id="pending-heading">{apartados.find((apartado) => apartado.id === vista)?.nombre}</h2>
          <p>Este apartado todavía no está disponible.</p>
        </section>
      )}
    </main>
  )
}

export default App
