import { useEffect, useState } from 'react'
import './App.css'

type ConnectionStatus = 'connecting' | 'connected' | 'error'

function App() {
  const [status, setStatus] = useState<ConnectionStatus>('connecting')

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

        setStatus('connected')
      } catch {
        if (!controller.signal.aborted) setStatus('error')
      }
    }

    void checkConnection()
    return () => controller.abort()
  }, [])

  const messages: Record<ConnectionStatus, string> = {
    connecting: 'Conectando con el backend…',
    connected: 'Backend conectado correctamente',
    error: 'Error de conexión con el backend',
  }

  return (
    <main className="connection-check">
      <h1>Prueba de conexión</h1>
      <p role="status" aria-live="polite">{messages[status]}</p>
    </main>
  )
}

export default App