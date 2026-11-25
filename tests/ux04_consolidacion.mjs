// Requiere Node 20.19+, Chromium o Chrome, frontend/node_modules y la .venv del proyecto.
// Desde la raíz: node --experimental-websocket tests/ux04_consolidacion.mjs
// Opciones: --visual (servidor local), --probar-fallo-arranque (prueba de limpieza).
// UX04_BROWSER y UX04_PYTHON permiten indicar ejecutables. UX04_CAPTURAS indica un directorio existente.
import assert from 'node:assert/strict'
import { execFile, spawn } from 'node:child_process'
import { once } from 'node:events'
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { createInterface } from 'node:readline'
import { tmpdir } from 'node:os'
import { basename, dirname, join, resolve } from 'node:path'
import { setTimeout as esperar } from 'node:timers/promises'
import { promisify } from 'node:util'
import { fileURLToPath } from 'node:url'
import { createServer as crearVite } from '../frontend/node_modules/vite/dist/node/index.js'

const ejecutarArchivo = promisify(execFile)
const raiz = fileURLToPath(new URL('../', import.meta.url))
const frontend = resolve(raiz, 'frontend')
const visual = process.argv.includes('--visual')
const lista = '/api/pronosticos/active/series'
const consolidacion = '/api/evaluaciones/consolidacion/active'

const falloArranque = process.argv.includes('--probar-fallo-arranque')
const PLAZO_ARRANQUE = 30_000
const PLAZO_CIERRE = 5_000

function conPlazo(promesa, milisegundos, operacion) {
  let temporizador
  const limite = new Promise((_, rechazar) => {
    temporizador = setTimeout(() => rechazar(new Error(`Tiempo agotado: ${operacion}.`)), milisegundos)
  })
  return Promise.race([promesa, limite]).finally(() => clearTimeout(temporizador))
}

function vigilarHijo(hijo, registrar, nombre, cerrarEntrada) {
  const fin = new Promise(resolver => {
    hijo.once('exit', resolver)
    hijo.once('error', () => { if (!hijo.pid) resolver() })
  })
  registrar(nombre, async () => {
    if (!hijo.pid || hijo.exitCode !== null || hijo.signalCode) return
    cerrarEntrada?.()
    if (await conPlazo(fin, PLAZO_CIERRE, `cerrar ${nombre}`).then(() => true, () => false)) return
    if (process.platform === 'win32') {
      await ejecutarArchivo('taskkill.exe', ['/PID', String(hijo.pid), '/T', '/F'], {
        windowsHide: true, timeout: PLAZO_CIERRE,
      }).catch(error => {
        if (hijo.exitCode === null && !hijo.signalCode) throw error
      })
    } else {
      try { process.kill(-hijo.pid, 'SIGKILL') } catch (error) {
        if (error.code !== 'ESRCH') throw error
      }
    }
    await conPlazo(fin, PLAZO_CIERRE, `confirmar el cierre de ${nombre}`)
  })
}

const ejecutarFastAPI = String.raw`
import json
import os
import sys
from backend.app.api import pronosticos
from backend.app.services import configuracion_experimentos as config
from tests.test_api_consolidacion_evaluaciones import ApiConsolidacionEvaluacionesTests

qa = ApiConsolidacionEvaluacionesTests()
try:
    qa.setUp()
    qa.app.include_router(pronosticos.router)
    datos = qa.datos
    variante = config.configuracion_predeterminada("random_forest")
    variante["parametros"]["n_estimators"] = 200
    datos.crear_grupo(3, "random_forest", efectiva=variante)
    datos.crear_grupo(4, config.BASE)
    datos.sql("UPDATE pronostico SET fecha_generacion=? WHERE id=401", ("2026-01-02T00:00:00",))
    datos.cambiar_registro(401, lambda r: r["versiones"].update(python="3.13.1"))
    entera = config.configuracion_predeterminada("random_forest")
    entera["parametros"]["max_features"] = 1
    datos.crear_grupo(5, "random_forest", efectiva=entera)
    datos.crear_grupo(6, "hist_gradient_boosting")

    print(json.dumps(["listo", os.getpid()]), flush=True)
    for linea in sys.stdin:
        numero, ruta = json.loads(linea)
        try:
            r = qa.cliente.get(ruta)
            print(json.dumps([numero, r.status_code, r.text]), flush=True)
        except Exception:
            print(json.dumps([numero, 500, '{"detail":{"codigo":"error_interno"}}']), flush=True)
finally:
    qa.doCleanups()
`
async function iniciarBackend(carpeta, registrar) {
  const python = process.env.UX04_PYTHON ?? (process.platform === 'win32' ?
    join(raiz, '.venv', 'Scripts', 'python.exe') : join(raiz, '.venv', 'bin', 'python'))
  const backend = spawn(python, ['-u', '-c', ejecutarFastAPI], {
    cwd: raiz, windowsHide: true, detached: process.platform !== 'win32', stdio: ['pipe', 'pipe', 'inherit'],
    env: {
      ...process.env, TMP: carpeta, TEMP: carpeta, TMPDIR: carpeta,
      PYTHONIOENCODING: 'utf-8', PYTHONDONTWRITEBYTECODE: '1',
    },
  })
  backend.stdin.on('error', () => { /* Cada escritura comunica su propio error. */ })
  vigilarHijo(backend, registrar, 'Python', () => backend.stdin.end())
  const lineas = createInterface({ input: backend.stdout })
  registrar('lector de Python', () => lineas.close())

  const pendientes = new Map()
  const noDisponible = [503, '{"detail":{"codigo":"base_no_disponible"}}']
  let siguiente = 0
  const inicio = new Promise((listo, fallo) => {
    lineas.once('line', linea => {
      try {
        const [estado, pid] = JSON.parse(linea)
        if (estado !== 'listo' || !Number.isInteger(pid)) throw new Error('Inicio inesperado del backend sintético.')
        listo(pid)
      } catch (error) { fallo(error) }
    })
    backend.once('error', fallo)
    backend.once('exit', codigo => fallo(new Error(`El backend sintético salió con código ${codigo}.`)))
  })
  backend.once('exit', () => {
    for (const completar of pendientes.values()) completar(noDisponible)
    pendientes.clear()
  })
  const pidApi = await conPlazo(inicio, PLAZO_ARRANQUE, 'iniciar FastAPI de prueba')
  lineas.on('line', linea => {
    const [numero, estado, cuerpo] = JSON.parse(linea)
    pendientes.get(numero)?.([estado, cuerpo])
  })

  function consultarApi(ruta) {
    const numero = ++siguiente
    const consulta = new Promise(completar => {
      pendientes.set(numero, completar)
      backend.stdin.write(JSON.stringify([numero, ruta]) + '\n', error => {
        if (error) completar(noDisponible)
      })
    })
    return conPlazo(consulta, PLAZO_ARRANQUE, 'consultar FastAPI de prueba')
      .catch(() => noDisponible).finally(() => pendientes.delete(numero))
  }
  return { consultarApi, pidApi }
}

const escenarios = {
  visual: [101, 102, 103, 201, 202, 203, 401, 601, 602],
  una: [101, 102, 103, 201, 202, 601],
  dos: [101, 102, 103, 201, 202, 203],
  'cambio-config': [101, 102, 103, 201, 202, 203],
  'resultado-distinto': [101, 102, 103, 201, 202, 203],
  tres: [101, 102, 103, 201, 202, 203, 301, 302, 303],
  tipo: [201, 202, 203, 501, 502, 503],
  parcial: [101, 102, 103, 201, 202, 203, 601],
  vacio: [],
}
const consultas = []
const peticionesNoGet = []
const tsx = '/src/__ux04_fixtures__.tsx'
const modulo = `
import { StrictMode, useState } from 'react'
import { createRoot } from 'react-dom/client'
import Consolidacion from './components/ConsolidacionEvaluaciones'
import './index.css'
import './App.css'

const casoPrueba = new URLSearchParams(location.search)
if (casoPrueba.has('sin-tipo') || casoPrueba.has('sin-tipo-resultado')) {
  const parse = JSON.parse
  JSON.parse = (text, reviver) => reviver && (casoPrueba.has('sin-tipo') || text.includes('"resultados"')) ? parse(text, function (key, value) {
    return reviver.call(this, key, value)
  }) : parse(text, reviver)
}

function EntornoQA() {
  const [otra, cambiar] = useState(false)
  const serie = otra ? { id: 84, store_nbr: 18, family: 'FAMILIA B' } : { id: 83, store_nbr: 17, family: 'FAMILIA A' }
  return <main className="app-page">
    <h2>Consolidación</h2>
    <p role="note"><strong>Entorno de prueba · datos sintéticos</strong><br />Tienda {serie.store_nbr} · {serie.family}</p>
    <button type="button" onClick={() => cambiar(!otra)}>Cambiar serie de prueba</button>
    <Consolidacion key={serie.id} seleccion={{ conjuntoId: 7, serie }} />
  </main>
}

createRoot(document.getElementById('root')).render(<StrictMode><EntornoQA /></StrictMode>)
`
let escenario = 'visual'
let enCurso = 0
let concurrencia = 0
async function iniciarWeb(cacheVite, consultarApi, registrar) {
  let vite
  let cerrandoVite = false
  let cierreVite
  const cerrarVite = () => cierreVite ??= vite.close()
  registrar('Vite', () => {
    cerrandoVite = true
    return vite && cerrarVite()
  })
  const inicioVite = crearVite({
    root: frontend, cacheDir: cacheVite, configFile: false, appType: 'custom',
    // El archivo de prueba es virtual. No se añade a frontend/src.
    plugins: [{
      name: 'ux04-fixtures',
      configureServer(servidor) { vite = servidor },
      resolveId(id) { if (id === tsx) return join(frontend, tsx.slice(1)).replaceAll('\\', '/') },
      load(id) { if (id.endsWith(tsx)) return modulo },
    }],
    oxc: { jsx: { runtime: 'automatic' } },
    server: { middlewareMode: true, ws: false, fs: { allow: [frontend, cacheVite] } },
  }).then(servidor => {
    vite = servidor
    if (cerrandoVite) return cerrarVite()
  })
  await conPlazo(inicioVite, PLAZO_ARRANQUE, 'iniciar Vite')
  const servidor = createServer(async (req, res) => {
    if (req.method !== 'GET') peticionesNoGet.push(req.method)
    const url = new URL(req.url, 'http://127.0.0.1')
    if (url.pathname === '/') {
      try {
        const html = await vite.transformIndexHtml('/', `<!doctype html><html lang="es"><meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>QA sintética · Consolidación</title><div id="root"></div><script type="module" src="${tsx}"></script></html>`)
        res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }).end(html)
      } catch (error) { res.writeHead(500).end(String(error)) }
      return
    }
    if (!url.pathname.startsWith('/api/')) {
      vite.middlewares(req, res, () => res.writeHead(404).end())
      return
    }
    const consulta = { metodo: req.method, ruta: url.pathname, grupos: url.searchParams.getAll('grupo') }
    consultas.push(consulta)
    const caso = escenario
    let respuesta = [404, '{}']
    const previa = /^\/api\/pronosticos\/active\/(\d+)\/configuracion$/.exec(url.pathname)
    if (req.method !== 'GET') respuesta = [405, '{}']
    else if (url.pathname === `${lista}/83` || url.pathname === `${lista}/84`) {
      respuesta = await consultarApi(url.pathname)
      if (respuesta[0] === 200 && url.pathname.endsWith('/83')) {
        const datos = JSON.parse(respuesta[1])
        datos.pronosticos = datos.pronosticos.filter(p => escenarios[caso].includes(p.id))
        respuesta = [200, JSON.stringify(datos)]
      }
    } else if (previa) {
      enCurso++
      concurrencia = Math.max(concurrencia, enCurso)
      try {
        await esperar(70)
        respuesta = caso === 'parcial' && previa[1] === '203' ?
          [503, '{"detail":{"codigo":"base_no_disponible"}}'] : await consultarApi(url.pathname)
        if (caso === 'cambio-config' && previa[1] === '201' && respuesta[0] === 200) {
          respuesta[1] = respuesta[1].replace(/"n_estimators":\s*100/, '"n_estimators":300')
        }
      } finally { enCurso-- }
    } else if (url.pathname === consolidacion) {
      respuesta = await consultarApi(url.pathname + url.search)
      if (caso === 'resultado-distinto' && respuesta[0] === 200) {
        respuesta[1] = respuesta[1].replace(/"n_estimators":\s*100/, '"n_estimators":300')
      }
    }
    res.writeHead(respuesta[0], { 'content-type': 'application/json', 'cache-control': 'no-store' }).end(respuesta[1])
  })
  const escucha = new AbortController()
  registrar('servidor HTTP', async () => {
    if (!servidor.listening) { escucha.abort(); return }
    const cerrado = new Promise((resolver, rechazar) => servidor.close(error => error ? rechazar(error) : resolver()))
    servidor.closeAllConnections()
    await cerrado
    escucha.abort()
  })
  servidor.listen({ port: 0, host: '127.0.0.1', signal: escucha.signal })
  await conPlazo(once(servidor, 'listening'), PLAZO_ARRANQUE, 'abrir el puerto HTTP')
  return `http://127.0.0.1:${servidor.address().port}`
}

async function navegador(carpeta, registrar) {
  const candidatos = [process.env.UX04_BROWSER, ...(
    process.platform === 'win32' ? [
      join(process.env.ProgramFiles ?? 'C:/Program Files', 'Google/Chrome/Application/chrome.exe'),
    ] : ['/usr/bin/chromium', '/usr/bin/google-chrome']
  )].filter(Boolean)
  for (const ejecutable of candidatos) {
    try { await access(ejecutable) } catch { continue }
    const proceso = spawn(ejecutable, ['--headless', '--no-first-run', '--disable-background-networking', '--disable-gpu',
      '--remote-debugging-port=0', `--user-data-dir=${carpeta}`, 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'], windowsHide: true, detached: process.platform !== 'win32' })
    vigilarHijo(proceso, registrar, 'Chromium')
    let salida = ''
    proceso.stderr.setEncoding('utf8').on('data', fragmento => { salida = (salida + fragmento).slice(-4000) })
    await conPlazo(once(proceso, 'spawn'), PLAZO_ARRANQUE, 'iniciar Chromium')
    return { proceso, salida: () => salida }
  }
  throw new Error('Indique la ruta de Chromium o Chrome con UX04_BROWSER.')
}

async function hasta(condicion, mensaje) {
  for (let i = 0; i < 160; i++) {
    const valor = await condicion()
    if (valor) return valor
    await esperar(50)
  }
  throw new Error(mensaje)
}

async function conectar(carpeta, salida, registrar) {
  const puerto = await hasta(async () => {
    const contenido = await readFile(join(carpeta, 'DevToolsActivePort'), 'utf8').catch(() => '')
    return contenido.split('\n')[0] || /DevTools listening on ws:\/\/127\.0\.0\.1:(\d+)/.exec(salida())?.[1]
  }, 'El navegador no publicó su puerto de pruebas.').catch(error => {
    throw new Error(`${error.message} ${salida()}`)
  })
  const paginas = await (await fetch(`http://127.0.0.1:${puerto}/json/list`, { signal: AbortSignal.timeout(PLAZO_CIERRE) })).json()
  const socket = new WebSocket(paginas.find(p => p.type === 'page').webSocketDebuggerUrl)
  registrar('conexión de Chromium', async () => {
    try {
      if (socket.readyState === WebSocket.OPEN) await enviar('Browser.close').catch(() => {})
    } finally {
      socket.close()
    }
  })
  let id = 0
  const pendientes = new Map()
  socket.addEventListener('message', ({ data }) => {
    const { id, result, error } = JSON.parse(data)
    const pendiente = pendientes.get(id)
    if (!pendiente) return
    pendientes.delete(id)
    if (error) pendiente.rechazar(new Error(error.message))
    else pendiente.resolver(result)
  })
  socket.addEventListener('close', () => {
    for (const { rechazar } of pendientes.values()) rechazar(new Error('Chromium cerró la conexión.'))
    pendientes.clear()
  })
  function enviar(method, params = {}) {
    const numero = ++id
    const solicitud = new Promise((resolver, rechazar) => {
      pendientes.set(numero, { resolver, rechazar })
      socket.send(JSON.stringify({ id: numero, method, params }))
    })
    return conPlazo(solicitud, PLAZO_CIERRE, method).finally(() => pendientes.delete(numero))
  }
  async function ejecutar(expression) {
    const r = await enviar('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text)
    return r.result.value
  }
  await conPlazo(new Promise((resolver, rechazar) => {
    socket.addEventListener('open', resolver, { once: true })
    socket.addEventListener('error', rechazar, { once: true })
  }), PLAZO_ARRANQUE, 'conectar con Chromium')
  return { enviar, ejecutar }
}

async function probar({ enviar, ejecutar }, origen) {
  const capturas = process.env.UX04_CAPTURAS
  async function capturar(nombre, selector) {
    if (!capturas) return
    const bloque = selector === '.consolidation-results' ? 'start' : 'center'
    await ejecutar(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView({ block: ${JSON.stringify(bloque)} })`)
    const r = await enviar('Page.captureScreenshot', { format: 'png' })
    await writeFile(join(capturas, nombre), Buffer.from(r.data, 'base64'))
  }
  await enviar('Emulation.setDeviceMetricsOverride', { width: 1000, height: 900, deviceScaleFactor: 1, mobile: false })
  const paso = '.consolidation-step'
  const avanzado = '.consolidation-advanced'
  const buscar = 'Buscar configuraciones para comparar'
  const preparar = 'Preparar pronósticos'
  const consultar = 'Consultar consolidación'
  const consultarDom = `texto => [...document.querySelectorAll('button')].find(b => b.textContent.trim() === texto)`
  const boton = texto => `(${consultarDom})(${JSON.stringify(texto)})`
  const disabled = texto => ejecutar(`${boton(texto)}?.disabled ?? true`)
  const pulsar = async texto => assert.equal(await ejecutar(`(() => {
    const b = ${boton(texto)}
    if (!b || b.disabled || !b.getClientRects().length) return false
    b.click()
    return true
  })()`), true, `No se pudo pulsar «${texto}»`)
  const texto = selector => ejecutar(`document.querySelector(${JSON.stringify(selector)})?.innerText ?? ''`)
  const hayTexto = async (selector, valor) => {
    try { await hasta(async () => (await texto(selector)).includes(valor), `Falta «${valor}» en ${selector}.`) }
    catch (error) { throw new Error(`${error.message}\n${await texto(selector)}`) }
  }
  const escoger = n => ejecutar(`document.querySelectorAll('.consolidation-choice input')[${n}].click()`)
  const cambiar = (selector, valor) => ejecutar(`(() => {
    const control = document.querySelector(${JSON.stringify(selector)})
    control.value = ${JSON.stringify(String(valor))}
    control.dispatchEvent(new Event('change', { bubbles: true }))
  })()`)
  const peticiones = ruta => consultas.filter(c => c.ruta === ruta)
  const abrir = async (caso, query = '') => {
    await hasta(() => enCurso === 0, 'Quedaron solicitudes pendientes del caso anterior.')
    escenario = caso
    consultas.length = 0
    concurrencia = 0
    await enviar('Page.navigate', { url: `${origen}/?caso=${caso}${query}` })
    await hasta(async () => (await ejecutar('location.search')) === `?caso=${caso}${query}` &&
      await ejecutar(`Boolean(${boton(buscar)})`), 'La página de prueba no montó el componente.')
    assert.ok(consultas.every(c => c.ruta === `${lista}/83`), 'El diagnóstico debe comenzar por acción explícita.')
    assert.equal(await ejecutar(`document.querySelector('${avanzado}').open`), false)
  }

  await abrir('una')
  await pulsar(buscar)
  await hayTexto(paso, '1 configuración identificada')
  await hayTexto(paso, '2 de 3 períodos disponibles')
  await hayTexto(paso, '1 de 3 períodos disponibles')
  await hayTexto(paso, 'Falta pronóstico')
  await hayTexto(paso, 'genere manualmente un pronóstico de validación en Pronósticos')
  await hayTexto(paso, 'conserve la técnica, las variables y los parámetros')
  assert.equal(await disabled(consultar), true)
  assert.ok(concurrencia <= 3)
  console.log('✓ Una configuración completa, faltantes y concurrencia acotada')

  await abrir('parcial')
  await pulsar(buscar)
  await hayTexto(paso, 'Búsqueda incompleta')
  await hayTexto(paso, 'Por verificar')
  assert.doesNotMatch(await texto(paso), /Insuficiencia comprobada para comparar dos/)
  const hgb = await ejecutar("[...document.querySelectorAll('article')].find(a => a.textContent.includes('HistGradientBoosting')).innerText")
  assert.match(hgb, /Falta pronóstico/)
  assert.doesNotMatch(hgb, /Por verificar/)
  console.log('✓ GET parcial, faltante comprobado e incertidumbre por técnica y período')

  await abrir('tipo')
  await pulsar(buscar)
  await hayTexto(paso, '2 configuraciones identificadas')
  await hayTexto(paso, 'Variables por división: 100')
  assert.match(await texto(paso), /Variables por división: 1(?: ·|\n)/)
  await abrir('tipo', '&sin-tipo=1')
  await pulsar(buscar)
  await hayTexto(paso, '0 configuraciones identificadas')
  await hayTexto(paso, 'Búsqueda incompleta')
  await ejecutar("document.querySelector('.evaluation-notice details').open = true")
  await hayTexto(paso, 'tipo numérico original')
  assert.equal(await disabled(consultar), true)
  console.log('✓ max_features entero y real permanecen separados, identidad indeterminada queda excluida')

  await abrir('resultado-distinto', '&sin-tipo-resultado=1')
  await pulsar(buscar)
  await hayTexto(paso, '2 configuraciones identificadas')
  await escoger(0)
  await escoger(1)
  await pulsar(preparar)
  await pulsar(consultar)
  await hayTexto('.consolidation-summary-cards', 'Árboles: 300')
  await hayTexto('.consolidation-summary-cards', 'Variables por división: tipo indeterminado')
  assert.doesNotMatch(await texto('.consolidation-summary-cards'), /Variables por división: 100/)
  console.log('✓ El resultado no hereda el tipo de max_features de otra configuración')

  await abrir('tres')
  await pulsar(buscar)
  await hayTexto(paso, '3 configuraciones identificadas')
  assert.equal(await ejecutar("document.querySelectorAll('.consolidation-choice input:checked').length"), 0)
  assert.match(await texto(paso), /Árboles: 100/)
  assert.match(await texto(paso), /Árboles: 200/)
  await escoger(2)
  await escoger(0)
  await escoger(1)
  assert.equal(await disabled(consultar), true)
  await pulsar(preparar)
  assert.equal(peticiones(consolidacion).length, 0)
  await pulsar(consultar)
  await hayTexto('.consolidation-results', '45/45 observaciones')
  assert.deepEqual(peticiones(consolidacion).at(-1).grupos, ['301,302,303', '101,102,103', '201,202,203'])
  await hayTexto('.consolidation-summary-cards', 'WAPE')
  assert.equal(await ejecutar("document.querySelectorAll('.consolidation-summary-card').length"), 3)
  await hayTexto('.consolidation-results', 'Detalle diario por corte')
  assert.equal(await ejecutar(`document.querySelector('${avanzado}').open`), false)
  console.log('✓ Tres configuraciones, incluidas dos de una técnica, con consulta válida y detalle')

  await abrir('dos')
  await pulsar(buscar)
  await pulsar('Reiniciar búsqueda')
  await pulsar('Cancelar análisis')
  await hayTexto(paso, 'Búsqueda cancelada')
  await esperar(200)
  assert.doesNotMatch(await texto(paso), /configuraciones identificadas/)
  await pulsar(buscar)
  await hayTexto(paso, '2 configuraciones identificadas')
  await escoger(0)
  await escoger(1)
  await pulsar('Cancelar selección')
  assert.equal(await disabled(preparar), true)
  await escoger(0)
  await escoger(1)
  await pulsar(preparar)
  await pulsar(consultar)
  await hayTexto('.consolidation-results', '45/45 observaciones')
  assert.deepEqual(peticiones(consolidacion).at(-1).grupos, ['101,102,103', '201,202,203'])
  await pulsar('Actualizar listado de pronósticos')
  await hasta(async () => await ejecutar(`Boolean(${boton(buscar)})`), 'El listado no terminó de actualizarse.')
  assert.equal(await disabled(consultar), true)
  await hayTexto(`${paso}:last-child`, 'Vuelva a buscar configuraciones')
  await pulsar(buscar)
  await hasta(async () => !(await disabled(consultar)), 'La preparación no volvió a estar disponible tras confirmarse.')
  assert.equal(await ejecutar("document.querySelectorAll('.consolidation-choice input:checked').length"), 2)
  escenario = 'cambio-config'
  await pulsar('Actualizar listado de pronósticos')
  await hasta(async () => await ejecutar(`Boolean(${boton(buscar)})`), 'El listado no terminó de actualizarse.')
  await pulsar(buscar)
  await hayTexto(`${paso}:last-child`, 'La búsqueda no pudo confirmar la selección anterior')
  assert.equal(await disabled(consultar), true)
  assert.doesNotMatch(await texto(`${paso}:last-child`), /Árboles: 100/)
  await pulsar(buscar)
  await pulsar('Cambiar serie de prueba')
  await hayTexto(paso, 'No hay pronósticos guardados')
  await esperar(200)
  assert.equal(await disabled(consultar), true)
  assert.equal(await ejecutar("Boolean(document.querySelector('.consolidation-results'))"), false)
  await pulsar('Cambiar serie de prueba')
  await hasta(async () => !(await disabled(buscar)), 'El listado no reapareció.')
  assert.equal(await disabled(consultar), true)
  console.log('✓ Dos configuraciones, cancelación y cambio de contexto sin resultados obsoletos')

  await abrir('visual')
  await ejecutar(`document.querySelector('${avanzado} > summary').click()`)
  await cambiar('.consolidation-count select', 3)
  assert.equal(await ejecutar("document.querySelectorAll('.consolidation-slot select').length"), 9)
  await cambiar('.consolidation-count select', 2)
  const primerControl = '.consolidation-group:first-child .consolidation-slot select'
  await cambiar(primerControl, 101)
  await pulsar('Inspeccionar configuración')
  await hayTexto('.consolidation-preview', 'Contrato / reglas')
  await pulsar('Actualizar listado de pronósticos')
  await hasta(async () => await ejecutar(`Boolean(${boton(buscar)})`), 'El listado no terminó de actualizarse.')
  assert.equal(await ejecutar(`document.querySelector('${primerControl}').value`), '101')
  assert.equal(await ejecutar(`document.querySelector('${avanzado}').open`), false)
  await pulsar(buscar)
  await hayTexto(paso, '2 configuraciones identificadas')
  await escoger(0)
  await escoger(1)
  assert.equal(await disabled(preparar), true)
  const candidato = '.consolidation-candidate select'
  await hayTexto(`${paso}:nth-of-type(2)`, 'comparten la configuración predictiva')
  await hayTexto(`${paso}:nth-of-type(2)`, 'La compatibilidad se comprobará al consultar la consolidación')
  assert.match(await ejecutar(`document.querySelector('${candidato}').innerText`), /2 de enero de 2026 .*pronóstico 401 · experimento 1401/)
  assert.equal(await ejecutar(`document.querySelector('${candidato}').value`), '')
  await cambiar(candidato, 401)
  await capturar('ux04-selector.png', '.consolidation-choice:has(.consolidation-candidate)')
  await pulsar(preparar)
  await pulsar('Conservar grupos actuales')
  assert.equal(await ejecutar(`document.querySelector('${primerControl}').value`), '101')
  assert.equal(await disabled(consultar), true)
  assert.equal(peticiones(consolidacion).length, 0)
  await escoger(0)
  await escoger(1)
  await cambiar(candidato, 401)
  await pulsar(preparar)
  await pulsar('Reemplazar y preparar pronósticos')
  assert.equal(await ejecutar(`document.querySelector('${primerControl}').value`), '401')
  assert.equal(peticiones(consolidacion).length, 0)
  await pulsar(consultar)
  const rechazo = `${paso}:last-child [role="alert"]`
  await hayTexto(rechazo, 'Los parámetros pueden coincidir aunque las versiones del software sean distintas o no se puedan verificar')
  await hayTexto(rechazo, 'pruebe otro candidato del mismo período')
  await hayTexto(rechazo, 'Se conservarán las demás elecciones')
  const detalleRechazo = `${paso}:last-child .evaluation-notice details`
  assert.equal(await ejecutar(`document.querySelector(${JSON.stringify(detalleRechazo)}).open`), false)
  assert.match(await ejecutar(`document.querySelector(${JSON.stringify(detalleRechazo)}).textContent`), /versiones_incompatibles/)
  await capturar('ux04-rechazo.png', `${paso}:last-child .evaluation-notice`)
  assert.equal(await ejecutar("document.querySelectorAll('.consolidation-choice input:checked:disabled').length"), 2)
  await pulsar('Editar selección rechazada')
  await hayTexto(`${paso}:last-child`, 'Hay una selección en edición')
  assert.equal(await disabled(consultar), true)
  assert.equal(peticiones(consolidacion).length, 1)
  assert.equal(await ejecutar(`document.querySelector('${candidato}').value`), '401')
  assert.equal(await ejecutar("document.querySelectorAll('.consolidation-choice input:checked').length"), 2)
  await cambiar(candidato, 101)
  await pulsar(preparar)
  await pulsar('Reemplazar y preparar pronósticos')
  await pulsar(consultar)
  await hayTexto('.consolidation-results', '45/45 observaciones')
  assert.deepEqual(peticiones(consolidacion).at(-1).grupos, ['101,102,103', '201,202,203'])
  assert.equal(await ejecutar(`document.querySelector('${avanzado}').open`), false)
  await capturar('ux04-recuperacion.png', '.consolidation-results')
  await enviar('Emulation.setDeviceMetricsOverride', { width: 360, height: 800, deviceScaleFactor: 1, mobile: true })
  assert.equal(await ejecutar('document.documentElement.scrollWidth <= innerWidth'), true, 'Desbordamiento horizontal a 360 px')
  assert.equal(await ejecutar("[...document.querySelectorAll('.consolidation-summary-card')].every(c => c.scrollWidth <= c.clientWidth)"), true)
  await capturar('ux04-resultados-movil.png', '.consolidation-results')
  console.log('✓ Candidatos múltiples, modo manual, conservación, confirmación, rechazo y recuperación')
}

async function ejecutarQA() {
  const limpieza = []
  const registrar = (nombre, cerrar) => limpieza.unshift({ nombre, cerrar })
  const interrupcion = new AbortController()
  const interrumpir = () => interrupcion.abort()
  process.once('SIGINT', interrumpir)
  process.once('SIGTERM', interrumpir)
  let errorPrincipal
  const erroresCierre = []
  try {
    const raizTemporal = resolve(tmpdir())
    const carpeta = await mkdtemp(join(raizTemporal, 'tesis-ux04-recursos-'))
    registrar('directorio temporal', async () => {
      if (dirname(carpeta) !== raizTemporal || !basename(carpeta).startsWith('tesis-ux04-recursos-')) {
        throw new Error('La ruta de limpieza está fuera del directorio temporal de QA.')
      }
      await rm(carpeta, { recursive: true, force: true, maxRetries: 6, retryDelay: 200 })
    })
    const cacheVite = join(carpeta, 'vite')
    const perfil = join(carpeta, 'chromium')
    const { pidApi, consultarApi } = await iniciarBackend(carpeta, registrar)
    const origen = await iniciarWeb(cacheVite, consultarApi, registrar)
    console.log(`QA sintética disponible en ${origen}`)
    if (visual) {
      console.log(`PID web: ${process.pid} · PID FastAPI de prueba: ${pidApi} · Puerto web: ${new URL(origen).port}`)
      console.log('Abra esa dirección en el navegador. Pulse Ctrl+C en esta consola para cerrar QA.')
      if (!interrupcion.signal.aborted) await once(interrupcion.signal, 'abort')
    } else {
      if (interrupcion.signal.aborted) throw new Error('QA interrumpida.')
      await mkdir(perfil)
      const { proceso, salida } = await navegador(perfil, registrar)
      const conexion = await conectar(perfil, salida, registrar)
      if (falloArranque) {
        console.log('Evidencia del fallo: ' + JSON.stringify({ origen, carpeta, pidApi, pidChromium: proceso.pid }))
        throw new Error('Fallo controlado durante el arranque de QA.')
      }
      await probar(conexion, origen)
      assert.deepEqual(peticionesNoGet, [])
      console.log('✓ Cero solicitudes POST durante todo el recorrido')
    }
  } catch (error) {
    errorPrincipal = error
  } finally {
    for (const { nombre, cerrar } of limpieza) {
      try { await conPlazo(Promise.resolve().then(cerrar), 4 * PLAZO_CIERRE, `limpiar ${nombre}`) }
      catch (error) { erroresCierre.push(new Error(`No se pudo limpiar ${nombre}.`, { cause: error })) }
    }
    process.removeListener('SIGINT', interrumpir)
    process.removeListener('SIGTERM', interrumpir)
  }
  if (erroresCierre.length) {
    console.error(new AggregateError(errorPrincipal ? [errorPrincipal, ...erroresCierre] : erroresCierre, 'Falló la limpieza de QA.'))
    process.exit(1)
  }
  console.log('Recursos propios de QA cerrados.')
  if (errorPrincipal) throw errorPrincipal
}

await ejecutarQA()
