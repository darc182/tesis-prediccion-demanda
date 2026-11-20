import { useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react'

const HISTORIAL = '/api/simulaciones/historial'
const MAX_SQLITE = 9223372036854775807n
const MAX_SEGURO = BigInt(Number.MAX_SAFE_INTEGER)
const IDS = [
  ['pronostico_id', 'ID de pronóstico'],
  ['conjunto_id', 'ID de conjunto'],
  ['serie_id', 'ID de serie'],
] as const
const INICIAL = { pronostico_id: '', conjunto_id: '', serie_id: '', modalidad: '', limite: '20' }
type Filtros = typeof INICIAL
type Consulta = { filtros: Filtros; desplazamiento: number }
type Estado<T> = { tipo: 'inicial' | 'cargando' } | { tipo: 'error'; mensaje: string } | { tipo: 'listo'; datos: T }

class ErrorHistorial extends Error {}

function verificar(condicion: unknown): asserts condicion {
  if (!condicion) throw new ErrorHistorial('La respuesta del historial contiene datos incoherentes.')
}

function objeto(valor: unknown): Record<string, unknown> {
  verificar(typeof valor === 'object' && valor !== null && !Array.isArray(valor))
  return valor as Record<string, unknown>
}

function texto(valor: unknown) {
  verificar(typeof valor === 'string' && valor.trim())
  return valor
}

function arreglo<T>(valor: unknown, leer: (dato: unknown, indice: number) => T) {
  verificar(Array.isArray(valor))
  return valor.map(leer)
}

function entero(valor: unknown, minimo: number, maximo = Number.MAX_SAFE_INTEGER) {
  verificar(typeof valor === 'number' && Number.isInteger(valor) && valor >= minimo)
  if (!Number.isSafeInteger(valor)) {
    throw new ErrorHistorial('La respuesta contiene un identificador o conteo que supera el rango entero seguro del navegador.')
  }
  verificar(valor <= maximo)
  return valor
}

function cantidad(valor: unknown, conSigno = false) {
  verificar(typeof valor === 'number' && Number.isFinite(valor) && (conSigno || valor >= 0))
  return valor
}

function fecha(valor: unknown) {
  const dia = texto(valor)
  verificar(/^\d{4}-\d{2}-\d{2}$/.test(dia))
  const instante = new Date(dia + 'T00:00:00Z')
  verificar(Number.isFinite(+instante) && instante.toISOString().slice(0, 10) === dia)
  return dia
}

function momentoUTC(valor: unknown) {
  const utc = texto(valor)
  verificar(/^\d{4}-\d{2}-\d{2}T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\dZ$/.test(utc))
  fecha(utc.slice(0, 10))
  return utc
}

function serie(valor: unknown) {
  const s = objeto(valor)
  return { id: entero(s.id, 1), tienda: entero(s.store_nbr, 1), familia: texto(s.family) }
}

const TECNICAS = {
  estacional_semanal_recursiva: 'Línea base semanal',
  random_forest: 'Random Forest',
  hist_gradient_boosting: 'HistGradientBoosting',
}

function identidad(d: Record<string, unknown>) {
  const tecnica = texto(d.tecnica)
  verificar(Object.hasOwn(TECNICAS, tecnica))
  return {
    pronostico: entero(d.pronostico_id, 1),
    experimento: d.experimento_id === null ? null : entero(d.experimento_id, 1),
    tecnica: TECNICAS[tecnica as keyof typeof TECNICAS],
    corte: fecha(d.fecha_corte),
  }
}

function cabecera(d: Record<string, unknown>) {
  verificar(d.modalidad === 'individual' || d.modalidad === 'comparacion')
  const numero = entero(d.numero_escenarios, 1, 3)
  verificar(d.modalidad === 'individual' ? numero === 1 : numero > 1)
  return { id: entero(d.ejecucion_id, 1), registro: momentoUTC(d.fecha_registro), modalidad: d.modalidad, numero }
}

function resumenListado(valor: unknown) {
  const d = objeto(valor)
  const c = cabecera(d)
  const etiquetas = arreglo(d.etiquetas, (e, i) => {
    verificar(e === 'ABC'[i])
    return e
  })
  verificar(etiquetas.length === c.numero)
  return { ...c, ...identidad(d), etiquetas, conjunto: entero(d.conjunto_id, 1), serie: serie(d.serie) }
}

function leerListado(valor: unknown, { filtros, desplazamiento }: Consulta) {
  const d = objeto(valor)
  const total = entero(d.total, 0)
  const limite = entero(d.limite, 1, 50)
  verificar(limite === Number(filtros.limite) && d.desplazamiento === desplazamiento)
  const filas = arreglo(d.ejecuciones, resumenListado)
  verificar(filas.length === Math.min(limite, Math.max(0, total - desplazamiento)))
  verificar(filas.every((f, i) =>
    (i === 0 || filas[i - 1].id > f.id) &&
    (!filtros.pronostico_id || f.pronostico === Number(filtros.pronostico_id)) &&
    (!filtros.conjunto_id || f.conjunto === Number(filtros.conjunto_id)) &&
    (!filtros.serie_id || f.serie.id === Number(filtros.serie_id)) &&
    (!filtros.modalidad || f.modalidad === filtros.modalidad)))
  return { total, limite, filas }
}

function parametros(d: Record<string, unknown>) {
  return {
    inventario: cantidad(d.inventario_disponible),
    plazo: entero(d.tiempo_abastecimiento, 0, 15),
    reserva: cantidad(d.reserva),
  }
}

function leerEscenario(valor: unknown, solicitado: unknown, indice: number, fechas: string[],
  estimaciones: number[], origen: ReturnType<typeof identidad>, regla: string) {
  const r = objeto(valor)
  const p = objeto(solicitado)
  const etiqueta = 'ABC'[indice]
  verificar(r.etiqueta === etiqueta && p.etiqueta === etiqueta)
  const s = objeto(r.simulacion)
  verificar(mismaIdentidad(identidad(s), origen) && s.horizonte === 15 && s.version_regla === regla)
  const original = parametros(p)
  const publicado = parametros(objeto(s.parametros))
  verificar(original.inventario === publicado.inventario && original.plazo === publicado.plazo && original.reserva === publicado.reserva)
  const periodo = objeto(s.periodo_cobertura)
  verificar(periodo.dias === 15 && periodo.fecha_inicio === fechas[0] && periodo.fecha_fin === fechas[14])
  verificar(typeof s.recepcion_dentro_horizonte === 'boolean')
  const trayectoria = arreglo(s.trayectoria, (valorDia, i) => {
    const d = objeto(valorDia)
    const paso = entero(d.paso, 1, 15)
    const dia = fecha(d.fecha_objetivo)
    const demanda = cantidad(d.demanda_pronosticada)
    verificar(paso === i + 1 && dia === fechas[i] && demanda === estimaciones[i])
    return { paso, dia, demanda, recepcion: cantidad(d.recepcion_simulada),
      balance: cantidad(d.balance_neto, true), disponible: cantidad(d.disponible_hipotetico),
      pendiente: cantidad(d.demanda_pendiente_actual) }
  })
  verificar(trayectoria.length === 15)
  const resumen = objeto(s.resumen)
  verificar(resumen.recepcion_dentro_horizonte === s.recepcion_dentro_horizonte)
  const primeraPendiente = resumen.fecha_primera_demanda_pendiente === null ? null : fecha(resumen.fecha_primera_demanda_pendiente)
  verificar(primeraPendiente === null || fechas.includes(primeraPendiente))
  return {
    etiqueta, parametros: original, trayectoria, demanda: cantidad(s.demanda_considerada),
    reposicion: cantidad(s.reposicion), recepcion: fecha(s.fecha_recepcion), dentro: s.recepcion_dentro_horizonte,
    supuestos: arreglo(s.supuestos, texto), advertencias: arreglo(s.advertencias, texto),
    primeraPendiente, mayorPendiente: cantidad(resumen.mayor_demanda_pendiente_actual),
    diasNegativos: entero(resumen.dias_con_balance_negativo, 0, 15),
    balanceFinal: cantidad(resumen.balance_neto_final, true),
    disponibleFinal: cantidad(resumen.disponible_hipotetico_final),
    pendienteFinal: cantidad(resumen.demanda_pendiente_final),
  }
}

function mismaIdentidad(a: ReturnType<typeof identidad>, b: ReturnType<typeof identidad>) {
  return a.pronostico === b.pronostico && a.experimento === b.experimento && a.tecnica === b.tecnica && a.corte === b.corte
}

function leerDetalle(valor: unknown, seleccion: ReturnType<typeof resumenListado>) {
  const d = objeto(valor)
  const c = cabecera(d)
  verificar(c.id === seleccion.id && c.modalidad === seleccion.modalidad && c.numero === seleccion.numero && c.registro === seleccion.registro)
  const h = objeto(d.historico)
  verificar(h.version_documento === 1)
  const regla = texto(h.version_regla)
  verificar(regla === 'cobertura_horizonte_15_pendientes_v1' &&
    h.version_politica_numerica === 'binary64_fraction_v1')
  const implementacion = texto(h.identificador_implementacion)
  const o = objeto(h.origen)
  const identidadOrigen = identidad(o)
  const serieOrigen = serie(o.serie)
  const conjunto = entero(o.conjunto_id, 1)
  verificar(mismaIdentidad(identidadOrigen, seleccion) && conjunto === seleccion.conjunto &&
    serieOrigen.id === seleccion.serie.id && serieOrigen.tienda === seleccion.serie.tienda && serieOrigen.familia === seleccion.serie.familia)
  const contexto = objeto(h.contexto)
  verificar(mismaIdentidad(identidad(contexto), identidadOrigen) && contexto.version_regla === regla &&
    o.horizonte === 15 && contexto.horizonte === 15)
  const fechas = arreglo(contexto.fechas_objetivo, fecha)
  const corteMs = Date.parse(identidadOrigen.corte + 'T00:00:00Z')
  verificar(fechas.length === 15 && fechas.every((dia, i) => Date.parse(dia + 'T00:00:00Z') === corteMs + (i + 1) * 86400000))
  const estimaciones = arreglo(o.detalles, (valorDia, i) => {
    const fila = objeto(valorDia)
    verificar(fila.paso === i + 1 && fila.fecha_objetivo === fechas[i])
    return cantidad(fila.venta_estimada)
  })
  verificar(estimaciones.length === 15)
  const solicitud = objeto(h.solicitud)
  verificar(solicitud.modalidad === c.modalidad && solicitud.pronostico_id === identidadOrigen.pronostico)
  const solicitados = arreglo(solicitud.escenarios, objeto)
  verificar(solicitados.length === c.numero)
  const escenarios = arreglo(h.resultados, (r, i) => leerEscenario(r, solicitados[i], i, fechas, estimaciones, identidadOrigen, regla))
  verificar(escenarios.length === c.numero)
  return { ...c, ...identidadOrigen, conjunto, serie: serieOrigen, fechas, regla, escenarios, implementacion,
    versionDocumento: h.version_documento, advertencias: arreglo(h.advertencias, texto) }
}

const ERRORES: Record<string, string> = {
  ejecucion_inexistente: 'La ejecución histórica solicitada ya no está disponible.',
  esquema_no_preparado: 'La base necesita una preparación explícita del esquema de simulaciones.',
  esquema_incompatible: 'El esquema de simulaciones de la base es incompatible.',
  version_no_soportada: 'La versión del esquema de simulaciones no está soportada.',
  snapshot_corrupto: 'El historial solicitado está incompleto o contiene datos inconsistentes.',
  snapshot_incompatible: 'La versión del historial solicitado no es compatible.',
  base_no_disponible: 'La base de datos del historial no está disponible.',
}

async function obtener(url: string, signal: AbortSignal): Promise<unknown> {
  let respuesta: Response
  try {
    respuesta = await fetch(url, { method: 'GET', cache: 'no-store', signal })
  } catch {
    throw new ErrorHistorial('No se pudo conectar para consultar el historial. Compruebe la conexión e inténtelo de nuevo.')
  }
  let datos: unknown
  try {
    datos = await respuesta.json()
  } catch {
    throw new ErrorHistorial('La API del historial devolvió un JSON inválido.')
  }
  if (!respuesta.ok) {
    const codigo = (datos as { detail?: { codigo?: unknown } } | null)?.detail?.codigo
    const mensaje = typeof codigo === 'string' && Object.hasOwn(ERRORES, codigo) ? ERRORES[codigo] : undefined
    throw new ErrorHistorial(mensaje ?? (respuesta.status === 404 ? ERRORES.ejecucion_inexistente :
      respuesta.status === 503 ? ERRORES.base_no_disponible : 'No se pudo completar la consulta del historial.'))
  }
  return datos
}

function mensajeError(error: unknown) {
  return error instanceof ErrorHistorial ? error.message : 'No se pudo completar la consulta del historial.'
}

function validarFiltros(filtros: Filtros) {
  for (const [clave, rotulo] of IDS) {
    const valor = filtros[clave]
    if (!valor) continue
    const digitos = valor.replace(/^0+/, '')
    if (!/^[0-9]+$/.test(valor) || !digitos || digitos.length > 19 || BigInt(digitos) > MAX_SQLITE) {
      return rotulo + ': ingrese un entero decimal ASCII positivo dentro del rango SQLite.'
    }
    if (BigInt(digitos) > MAX_SEGURO) return rotulo + ': supera el rango entero seguro del navegador.'
  }
  if (!/^[0-9]+$/.test(filtros.limite) || Number(filtros.limite) < 1 || Number(filtros.limite) > 50) {
    return 'El límite de ejecuciones por página debe ser un entero entre 1 y 50.'
  }
  if (!['', 'individual', 'comparacion'].includes(filtros.modalidad)) return 'Seleccione una modalidad válida.'
  return ''
}

function numero(valor: number) {
  if (Number.isInteger(valor) && !Number.isSafeInteger(valor)) return 'Fuera del rango seguro'
  return Object.is(valor, -0) ? '-0.0' : String(valor)
}

function nombreModalidad(modalidad: string) {
  return modalidad === 'individual' ? 'Individual' : 'Comparación'
}

function Datos({ campos }: { campos: [string, ReactNode][] }) {
  return <dl className="simulaciones-datos">{campos.map(([nombre, valor]) =>
    <div key={nombre}><dt>{nombre}</dt><dd>{valor}</dd></div>)}</dl>
}

function Notas({ titulo, notas }: { titulo: string; notas: string[] }) {
  if (!notas.length) return null
  return <div className="simulaciones-notas"><p><strong>{titulo}</strong></p>
    <ul>{notas.map((nota, i) => <li key={i}>{nota}</li>)}</ul></div>
}

function Detalle({ datos }: { datos: ReturnType<typeof leerDetalle> }) {
  return <section className="simulaciones-detalle" aria-labelledby="simulaciones-detalle">
    <h3 id="simulaciones-detalle">Ejecución {datos.id}</h3>
    <Datos campos={[
      ['Registro UTC', <time dateTime={datos.registro}>{datos.registro}</time>],
      ['Modalidad', nombreModalidad(datos.modalidad)], ['Escenarios', datos.numero],
      ['Conjunto', datos.conjunto], ['Serie', datos.serie.id], ['Tienda', datos.serie.tienda], ['Familia', datos.serie.familia],
      ['Pronóstico', datos.pronostico], ['Experimento', datos.experimento ?? 'Sin identificador'],
      ['Técnica', datos.tecnica], ['Fecha de corte', datos.corte], ['Horizonte', '15 días'],
      ['Regla histórica', datos.regla], ['Versión del documento', datos.versionDocumento],
      ['Política numérica', 'binary64_fraction_v1'], ['Implementación', datos.implementacion],
      ['Fechas consideradas', datos.fechas[0] + ' a ' + datos.fechas[14]],
    ]} />
    <Notas titulo="Advertencias de la ejecución" notas={datos.advertencias} />
    <p className="help-text">Las cantidades enteras que exceden el rango seguro del navegador se omiten. Los valores se muestran con la precisión numérica disponible en el navegador.</p>
    {datos.escenarios.map(e => <section className="simulaciones-escenario" key={e.etiqueta} aria-label={'Escenario ' + e.etiqueta}>
      <h4>Escenario {e.etiqueta}</h4>
      <Datos campos={[
        ['Inventario hipotético inicial', numero(e.parametros.inventario)], ['Plazo de abastecimiento', e.parametros.plazo + ' días'],
        ['Reserva', numero(e.parametros.reserva)], ['Demanda considerada', numero(e.demanda)],
        ['Reposición Q', numero(e.reposicion)], ['Fecha de recepción', e.recepcion],
        ['Recepción dentro del horizonte', e.dentro ? 'Sí' : 'No'],
      ]} />
      <h5>Resumen final</h5>
      <Datos campos={[
        ['Balance neto final', numero(e.balanceFinal)], ['Disponible hipotético final', numero(e.disponibleFinal)],
        ['Demanda pendiente final', numero(e.pendienteFinal)], ['Mayor demanda pendiente', numero(e.mayorPendiente)],
        ['Primera fecha con demanda pendiente', e.primeraPendiente ?? 'Sin demanda pendiente'],
        ['Días con balance negativo', e.diasNegativos],
      ]} />
      <Notas titulo="Supuestos" notas={e.supuestos} />
      <Notas titulo="Advertencias del escenario" notas={e.advertencias} />
      <div className="simulaciones-tabla" tabIndex={0} role="region" aria-label={'Trayectoria desplazable del escenario ' + e.etiqueta}>
        <table>
          <caption>Trayectoria de 15 días · Escenario {e.etiqueta}</caption>
          <thead><tr>{['Paso', 'Fecha', 'Demanda pronosticada', 'Recepción', 'Balance neto', 'Disponible hipotético', 'Demanda pendiente']
            .map(label => <th key={label} scope="col">{label}</th>)}</tr></thead>
          <tbody>{e.trayectoria.map(d => <tr key={d.paso}>
            <td>{d.paso}</td><th scope="row">{d.dia}</th><td>{numero(d.demanda)}</td><td>{numero(d.recepcion)}</td>
            <td>{numero(d.balance)}</td><td>{numero(d.disponible)}</td><td>{numero(d.pendiente)}</td>
          </tr>)}</tbody>
        </table>
      </div>
    </section>)}
  </section>
}

export default function Simulaciones() {
  const [borrador, setBorrador] = useState<Filtros>(INICIAL)
  const [consulta, setConsulta] = useState<Consulta>({ filtros: INICIAL, desplazamiento: 0 })
  const [listado, setListado] = useState<Estado<ReturnType<typeof leerListado>>>({ tipo: 'cargando' })
  const [detalle, setDetalle] = useState<Estado<ReturnType<typeof leerDetalle>>>({ tipo: 'inicial' })
  const [seleccion, setSeleccion] = useState<number | null>(null)
  const [errorFiltros, setErrorFiltros] = useState('')
  const peticionLista = useRef<AbortController | null>(null)
  const peticionDetalle = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    peticionLista.current = controller
    const vigente = () => peticionLista.current === controller && !controller.signal.aborted
    const query = new URLSearchParams({ limite: String(Number(consulta.filtros.limite)), desplazamiento: String(consulta.desplazamiento) })
    for (const [clave, valor] of Object.entries(consulta.filtros)) {
      if (clave !== 'limite' && valor) query.set(clave, clave === 'modalidad' ? valor : valor.replace(/^0+/, ''))
    }
    void obtener(HISTORIAL + '?' + query, controller.signal)
      .then(valor => {
        if (vigente()) setListado({ tipo: 'listo', datos: leerListado(valor, consulta) })
      })
      .catch(error => { if (vigente()) setListado({ tipo: 'error', mensaje: mensajeError(error) }) })
    return () => controller.abort()
  }, [consulta])

  useEffect(() => () => peticionDetalle.current?.abort(), [])

  function actualizar(filtros: Filtros, desplazamiento: number) {
    peticionLista.current?.abort()
    peticionDetalle.current?.abort()
    setSeleccion(null)
    setDetalle({ tipo: 'inicial' })
    setListado({ tipo: 'cargando' })
    setConsulta({ filtros, desplazamiento })
  }

  function aplicar(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const error = validarFiltros(borrador)
    setErrorFiltros(error)
    if (!error) actualizar(borrador, 0)
  }

  async function abrir(fila: ReturnType<typeof resumenListado>) {
    peticionDetalle.current?.abort()
    const controller = new AbortController()
    peticionDetalle.current = controller
    const vigente = () => peticionDetalle.current === controller && !controller.signal.aborted
    setSeleccion(fila.id)
    setDetalle({ tipo: 'cargando' })
    try {
      const valor = await obtener(HISTORIAL + '/' + fila.id, controller.signal)
      if (vigente()) setDetalle({ tipo: 'listo', datos: leerDetalle(valor, fila) })
    } catch (error) {
      if (vigente()) setDetalle({ tipo: 'error', mensaje: mensajeError(error) })
    }
  }

  const pagina = Math.floor(consulta.desplazamiento / Number(consulta.filtros.limite)) + 1
  return <section className="simulaciones-card" aria-labelledby="simulaciones-heading">
    <h2 id="simulaciones-heading">Historial de simulaciones</h2>
    <p>Los resultados corresponden a escenarios hipotéticos, no a inventario observado ni a órdenes reales de reposición.</p>
    <form onSubmit={aplicar}>
      <p id="simulaciones-filtros-ayuda" className="help-text">Los filtros de ID son opcionales. Use dígitos del 0 al 9 e identificadores dentro del rango entero seguro del navegador.</p>
      <div className="simulaciones-filtros" aria-describedby="simulaciones-filtros-ayuda">
        {IDS.map(([clave, rotulo]) => <label key={clave}>{rotulo}
          <input inputMode="numeric" value={borrador[clave]} onChange={e => setBorrador({ ...borrador, [clave]: e.target.value })} />
        </label>)}
        <label>Modalidad
          <select value={borrador.modalidad} onChange={e => setBorrador({ ...borrador, modalidad: e.target.value })}>
            <option value="">Todas</option><option value="individual">Individual</option><option value="comparacion">Comparación</option>
          </select>
        </label>
        <label>Ejecuciones por página
          <input inputMode="numeric" value={borrador.limite} onChange={e => setBorrador({ ...borrador, limite: e.target.value })} />
        </label>
      </div>
      <div className="simulaciones-acciones">
        <button type="submit">Aplicar filtros</button>
        <button type="button" onClick={() => actualizar(consulta.filtros, consulta.desplazamiento)}>Actualizar listado</button>
      </div>
      {errorFiltros && <p role="alert">{errorFiltros}</p>}
    </form>
    {listado.tipo === 'cargando' && <p role="status">Cargando listado de simulaciones…</p>}
    {listado.tipo === 'error' && <p role="alert">{listado.mensaje}</p>}
    {listado.tipo === 'listo' && <>
      <p role="status">Página {pagina} · {listado.datos.total} coincidencias</p>
      {listado.datos.filas.length === 0 ? <p role="status">
        {listado.datos.total === 0 ? 'No hay simulaciones para estos filtros.' : 'Esta página ya no contiene ejecuciones.'}
      </p> : <div className="simulaciones-tabla" tabIndex={0} role="region" aria-label="Listado histórico desplazable">
        <table>
          <caption>Ejecuciones de simulación</caption>
          <thead><tr>{['Ejecución', 'Registro UTC', 'Modalidad', 'Escenarios', 'Tienda · Familia', 'Técnica', 'Fecha de corte', 'Pronóstico', 'Detalle']
            .map(label => <th scope="col" key={label}>{label}</th>)}</tr></thead>
          <tbody>{listado.datos.filas.map(f => <tr key={f.id}>
            <th scope="row">{f.id}</th><td><time dateTime={f.registro}>{f.registro}</time></td><td>{nombreModalidad(f.modalidad)}</td>
            <td>{f.numero} · {f.etiquetas.join('/')}</td><td>{f.serie.tienda} · {f.serie.familia}</td><td>{f.tecnica}</td>
            <td>{f.corte}</td><td>{f.pronostico}</td>
            <td><button type="button" aria-label={'Ver detalle de la ejecución ' + f.id}
              aria-pressed={seleccion === f.id} onClick={() => { void abrir(f) }}>Ver detalle</button></td>
          </tr>)}</tbody>
        </table>
      </div>}
      <nav className="simulaciones-acciones" aria-label="Páginas del historial">
        <button type="button" disabled={consulta.desplazamiento === 0}
          onClick={() => actualizar(consulta.filtros, Math.max(0, consulta.desplazamiento - listado.datos.limite))}>Anterior</button>
        <button type="button" disabled={listado.datos.total - consulta.desplazamiento <= listado.datos.limite}
          onClick={() => actualizar(consulta.filtros, consulta.desplazamiento + listado.datos.limite)}>Siguiente</button>
      </nav>
    </>}
    {listado.tipo === 'listo' && listado.datos.filas.length > 0 && detalle.tipo === 'inicial' &&
      <p role="status">Seleccione una ejecución para consultar su detalle.</p>}
    {detalle.tipo === 'cargando' && <p role="status">Cargando detalle de la ejecución {seleccion}…</p>}
    {detalle.tipo === 'error' && <p role="alert">{detalle.mensaje}</p>}
    {detalle.tipo === 'listo' && <Detalle datos={detalle.datos} />}
  </section>
}
