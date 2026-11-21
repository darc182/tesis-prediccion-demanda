import { useEffect, useImperativeHandle, useRef, useState, type ComponentProps, type FormEvent, type ReactNode, type Ref } from 'react'
import SelectorSerie, { type SerieTemporal } from './SelectorSerie'

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

function Detalle({ datos, prefijo = 'simulaciones-detalle' }: { datos: ReturnType<typeof leerDetalle>; prefijo?: string }) {
  return <section className="simulaciones-detalle" aria-labelledby={prefijo}>
    <h3 id={prefijo}>Ejecución {datos.id}</h3>
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

type ControlHistorial = {
  refrescar: () => void
  abrir: (fila: ReturnType<typeof resumenListado>) => Promise<void>
}

function Historial({ control }: { control: Ref<ControlHistorial> }) {
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

  useImperativeHandle(control, () => ({
    refrescar: () => actualizar(consulta.filtros, consulta.desplazamiento),
    abrir,
  }))

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

type SeleccionActiva = { conjuntoId: number; serie: SerieTemporal }
type Pronostico = ReturnType<typeof identidad>
type Modalidad = 'individual' | 'comparacion'
type Campos = { inventario_disponible: string; tiempo_abastecimiento: string; reserva: string }
type Intencion = Readonly<{
  url: string
  cuerpo: string
  contexto: string
  conjunto: number
  serie: ReturnType<typeof serie>
  pronostico: Pronostico
  modalidad: Modalidad
  escenarios: readonly Readonly<Campos>[]
}>
type Envio =
  | { tipo: 'borrador' }
  | { tipo: 'enviando' | 'desconocido'; intencion: Intencion; mensaje: string }
  | { tipo: 'error' | 'conflicto'; intencion: Intencion; mensaje: string; http: number;
      reintento: boolean; conservar: boolean }

const CAMPOS: [keyof Campos, string][] = [
  ['inventario_disponible', 'Inventario hipotético inicial'],
  ['tiempo_abastecimiento', 'Plazo de abastecimiento (días)'],
  ['reserva', 'Reserva hipotética'],
]
const VACIO: Campos = { inventario_disponible: '', tiempo_abastecimiento: '', reserva: '' }
const CORTES_VALIDACION = ['2017-06-16', '2017-07-01', '2017-07-16']
const LEXEMA_JSON = /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$/
const MAX_LEXEMA = 64

function claveContexto(s: SeleccionActiva) {
  return JSON.stringify([s.conjuntoId, s.serie.id, s.serie.store_nbr, s.serie.family])
}

function leerPronosticos(valor: unknown, seleccion: SeleccionActiva) {
  const d = objeto(valor)
  const origen = serie(d.serie)
  verificar(d.conjunto_id === seleccion.conjuntoId && origen.id === seleccion.serie.id &&
    origen.tienda === seleccion.serie.store_nbr && origen.familia === seleccion.serie.family)
  verificar(Array.isArray(d.pronosticos))
  const vistos = new Set<number>()
  const elegibles: Pronostico[] = []
  for (const item of d.pronosticos) {
    const p = objeto(item)
    if (p.tipo !== 'validacion' || p.horizonte !== 15 ||
      typeof p.fecha_corte !== 'string' || !CORTES_VALIDACION.includes(p.fecha_corte) ||
      typeof p.tecnica !== 'string' || !Object.hasOwn(TECNICAS, p.tecnica) ||
      typeof p.id !== 'number' || !Number.isSafeInteger(p.id) || p.id <= 0 ||
      typeof p.experimento_id !== 'number' || !Number.isSafeInteger(p.experimento_id) || p.experimento_id <= 0) continue
    verificar(!vistos.has(p.id))
    vistos.add(p.id)
    elegibles.push(identidad({ ...p, pronostico_id: p.id }))
  }
  return elegibles
}

function cantidadJSON(token: string) {
  if (token.length > MAX_LEXEMA || !LEXEMA_JSON.test(token) || token !== token.trim() || token === '-0') return false
  const valor = Number(token)
  const enteroLiteral = !/[.eE]/.test(token)
  return Number.isFinite(valor) && valor >= 0 &&
    (!enteroLiteral || Number.isSafeInteger(valor)) &&
    (valor !== 0 || !/[1-9]/.test(token.split(/[eE]/)[0]))
}

function datosEscenario(campos: Readonly<Campos>, i: number) {
  const etiqueta = 'ABC'[i]
  const { inventario_disponible: inventario, tiempo_abastecimiento: plazo, reserva } = campos
  if (!cantidadJSON(inventario)) throw new ErrorHistorial('Escenario ' + etiqueta + ' · ingrese un inventario válido.')
  if (!cantidadJSON(reserva)) throw new ErrorHistorial('Escenario ' + etiqueta + ' · ingrese una reserva válida.')
  if (!/^(?:[0-9]|1[0-5])$/.test(plazo) || plazo !== plazo.trim()) {
    throw new ErrorHistorial('Escenario ' + etiqueta + ' · el plazo debe ser un entero entre 0 y 15.')
  }
  return '"inventario_disponible":' + inventario + ',"tiempo_abastecimiento":' + plazo + ',"reserva":' + reserva
}

function construirCuerpo(pronostico: number, modalidad: Modalidad, escenarios: readonly Readonly<Campos>[], clave: string) {
  verificar(Number.isSafeInteger(pronostico) && pronostico > 0)
  verificar(modalidad === 'individual' ? escenarios.length === 1 : modalidad === 'comparacion' && [2, 3].includes(escenarios.length))
  const campos = escenarios.map(datosEscenario)
  const parametros = modalidad === 'individual' ? campos[0] :
    '"escenarios":[' + campos.map((item, i) => '{"etiqueta":"' + 'ABC'[i] + '",' + item + '}').join(',') + ']'
  const cuerpo = '{"pronostico_id":' + pronostico + ',' + parametros + ',"clave_idempotencia":' + JSON.stringify(clave) + '}'
  verificar(new TextEncoder().encode(cuerpo).length <= 16384)
  return cuerpo
}

function nuevaIntencion(s: SeleccionActiva, pronostico: Pronostico, modalidad: Modalidad, campos: Campos[]): Intencion {
  // Validar los lexemas antes de generar la clave y conservar la misma cadena para cada reintento.
  campos.forEach(datosEscenario)
  if (typeof globalThis.crypto?.randomUUID !== 'function') {
    throw new ErrorHistorial('Este navegador necesita un contexto seguro para generar una clave de registro.')
  }
  const escenarios = Object.freeze(campos.map(campo => Object.freeze({ ...campo })))
  const cuerpo = construirCuerpo(pronostico.pronostico, modalidad, escenarios, crypto.randomUUID())
  return Object.freeze({
    url: '/api/simulaciones/' + modalidad, cuerpo, contexto: claveContexto(s), conjunto: s.conjuntoId,
    serie: Object.freeze(serie(s.serie)), pronostico: Object.freeze({ ...pronostico }), modalidad, escenarios,
  })
}

function leerConfirmacion(valor: unknown, status: number, intento: Intencion) {
  const d = objeto(valor)
  const creado = status === 201
  verificar((creado || status === 200) && d.estado_registro === (creado ? 'creado' : 'reutilizado') && d.hubo_escritura === creado)
  const cab = cabecera(d)
  verificar(cab.modalidad === intento.modalidad && cab.numero === intento.escenarios.length)
  const fila = { ...cab, ...intento.pronostico, conjunto: intento.conjunto,
    serie: intento.serie, etiquetas: Array.from('ABC'.slice(0, cab.numero)) }
  const datos = leerDetalle(d, fila)
  verificar(datos.escenarios.every((e, i) => {
    const original = intento.escenarios[i]
    return Object.is(e.parametros.inventario, Number(original.inventario_disponible)) &&
      e.parametros.plazo === Number(original.tiempo_abastecimiento) &&
      Object.is(e.parametros.reserva, Number(original.reserva))
  }))
  return { creado, fila, datos, contexto: intento.contexto }
}

const RECHAZOS: Record<string, [number, string]> = {
  entrada_invalida: [422, 'La API rechazó el formato de la solicitud.'],
  solicitud_invalida: [422, 'La solicitud tiene campos inválidos.'],
  pronostico_invalido: [422, 'El identificador del pronóstico es inválido.'],
  clave_idempotencia_invalida: [422, 'La API rechazó la identificación de la solicitud.'],
  escenarios_invalidos: [422, 'La API rechazó los escenarios seleccionados.'],
  parametros_invalidos: [422, 'Revise el inventario, el plazo y la reserva.'],
  precision_no_representable: [422, 'Una cantidad excede la precisión admitida.'],
  tipo_no_evaluable: [422, 'Seleccione un pronóstico de validación.'],
  corte_no_autorizado: [422, 'El corte de este pronóstico no está autorizado.'],
  pronostico_inexistente: [404, 'El pronóstico seleccionado ya no existe.'],
  pronostico_inactivo: [404, 'El pronóstico ya no pertenece al conjunto activo.'],
  conflicto_idempotencia: [409, 'La solicitud en conflicto corresponde a otro contenido.'],
  sin_conjunto_activo: [409, 'No hay un conjunto de datos activo.'],
  pronostico_corrupto: [409, 'Los datos del pronóstico presentan inconsistencias.'],
  resultados_incompatibles: [409, 'La API detectó resultados incompatibles.'],
  resultado_no_finito: [409, 'El resultado excede el rango numérico admitido.'],
  origen_modificado: [409, 'El pronóstico cambió durante el registro. Actualice la selección.'],
  snapshot_corrupto: [409, ERRORES.snapshot_corrupto],
  snapshot_incompatible: [409, ERRORES.snapshot_incompatible],
  esquema_no_preparado: [409, ERRORES.esquema_no_preparado],
  esquema_incompatible: [409, ERRORES.esquema_incompatible],
  version_no_soportada: [409, ERRORES.version_no_soportada],
  base_no_disponible: [503, ERRORES.base_no_disponible],
  error_persistencia: [503, 'La API informó que no pudo confirmar el registro.'],
}

async function enviar(intencion: Intencion) {
  let respuesta: Response
  try {
    respuesta = await fetch(intencion.url, {
      method: 'POST', cache: 'no-store', redirect: 'error',
      headers: { 'Content-Type': 'application/json' }, body: intencion.cuerpo,
    })
  } catch {
    return { tipo: 'desconocido', mensaje: 'Error de red durante el envío.' } as const
  }
  let valor: unknown
  try {
    valor = await respuesta.json()
  } catch {
    return { tipo: 'desconocido', mensaje: 'La API devolvió una respuesta que no se pudo leer.' } as const
  }
  if (respuesta.ok) {
    try {
      return { tipo: 'confirmado', registro: leerConfirmacion(valor, respuesta.status, intencion) } as const
    } catch {
      return { tipo: 'desconocido', mensaje: 'La API devolvió una confirmación incoherente.' } as const
    }
  }
  const codigo = (valor as { detail?: { codigo?: unknown } } | null)?.detail?.codigo
  const rechazo = typeof codigo === 'string' && Object.hasOwn(RECHAZOS, codigo) ? RECHAZOS[codigo] : null
  if (rechazo?.[0] === respuesta.status) {
    return {
      tipo: codigo === 'conflicto_idempotencia' ? 'conflicto' : 'error',
      mensaje: rechazo[1], http: respuesta.status, persistenciaSinConfirmar: codigo === 'error_persistencia',
    } as const
  }
  const mensaje = respuesta.status === 500 && codigo === 'error_interno'
    ? 'La API informó un error interno.' : 'La API devolvió una respuesta HTTP sin confirmación válida.'
  return { tipo: 'desconocido', mensaje } as const
}

function FormularioSimulacion({ seleccion, bloqueado, registrar }: {
  seleccion: SeleccionActiva; bloqueado: boolean; registrar: (intencion: Intencion) => void
}) {
  const [lista, setLista] = useState<Estado<Pronostico[]>>({ tipo: 'cargando' })
  const [revision, setRevision] = useState(0)
  const [id, setId] = useState('')
  const [modo, setModo] = useState<Modalidad>('individual')
  const [tercero, setTercero] = useState(false)
  const [campos, setCampos] = useState<Campos[]>([{ ...VACIO }, { ...VACIO }, { ...VACIO }])
  const [error, setError] = useState('')
  const { conjuntoId, serie: { id: serieId, store_nbr, family } } = seleccion

  useEffect(() => {
    const controlador = new AbortController()
    const esperado = { conjuntoId, serie: { id: serieId, store_nbr, family } }
    void obtener('/api/pronosticos/active/series/' + serieId, controlador.signal)
      .then(valor => {
        if (!controlador.signal.aborted) setLista({ tipo: 'listo', datos: leerPronosticos(valor, esperado) })
      })
      .catch(() => {
        if (!controlador.signal.aborted) setLista({ tipo: 'error', mensaje: 'No se pudo consultar el listado de pronósticos de esta serie.' })
      })
    return () => controlador.abort()
  }, [conjuntoId, serieId, store_nbr, family, revision])

  const pronostico = lista.tipo === 'listo' ? lista.datos.find(p => String(p.pronostico) === id) : undefined
  const numeroEscenarios = modo === 'individual' ? 1 : tercero ? 3 : 2

  function guardar(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (bloqueado || !pronostico) return
    setError('')
    try {
      registrar(nuevaIntencion(seleccion, pronostico, modo, campos.slice(0, numeroEscenarios)))
    } catch (error) {
      setError(error instanceof ErrorHistorial ? error.message : 'No se pudo preparar el registro.')
    }
  }

  return <>
    {lista.tipo === 'cargando' && <p role="status">Cargando pronósticos de la serie…</p>}
    {lista.tipo === 'error' && <p role="alert">{lista.mensaje}</p>}
    {lista.tipo === 'listo' && lista.datos.length === 0 && <p role="status">Esta serie no tiene pronósticos elegibles de validación.</p>}
    <button type="button" disabled={lista.tipo === 'cargando' || bloqueado} onClick={() => {
      setId(''); setError(''); setLista({ tipo: 'cargando' }); setRevision(n => n + 1)
    }}>Actualizar pronósticos</button>
    {lista.tipo === 'listo' && lista.datos.length > 0 && <form className="simulaciones-formulario" onSubmit={guardar} noValidate>
      <fieldset disabled={bloqueado}>
        <legend>Pronóstico y modalidad</legend>
        <label>Pronóstico de validación
          <select value={id} onChange={e => { setId(e.target.value); setError('') }}>
            <option value="">Seleccione un pronóstico</option>
            {lista.datos.map(p => <option value={p.pronostico} key={p.pronostico}>
              {p.tecnica} · {p.corte} · 15 días · Pronóstico {p.pronostico} · Experimento {p.experimento}
            </option>)}
          </select>
        </label>
        <label>Modalidad de creación
          <select value={modo} onChange={e => { setModo(e.target.value === 'comparacion' ? 'comparacion' : 'individual'); setError('') }}>
            <option value="individual">Individual A</option><option value="comparacion">Comparación A/B</option>
          </select>
        </label>
        {modo === 'comparacion' && <label className="simulaciones-casilla">
          <input type="checkbox" checked={tercero} onChange={e => { setTercero(e.target.checked); setError('') }} /> Incluir escenario C
        </label>}
      </fieldset>
      {campos.slice(0, numeroEscenarios).map((escenario, i) =>
        <fieldset disabled={bloqueado} key={i} aria-describedby="simulaciones-parametros-ayuda">
          <legend>Escenario {'ABC'[i]}</legend>
          <div className="simulaciones-filtros">
            {CAMPOS.map(([campo, rotulo]) => <label key={campo}>{rotulo}
              <input type="text" inputMode={campo === 'tiempo_abastecimiento' ? 'numeric' : 'text'}
                value={escenario[campo]} onChange={e => {
                  const valor = e.target.value
                  setCampos(anteriores => anteriores.map((x, j) => j === i ? { ...x, [campo]: valor } : x))
                  setError('')
                }} />
            </label>)}
          </div>
        </fieldset>)}
      {error && <p role="alert">{error}</p>}
      <button type="submit" disabled={!pronostico || bloqueado}>Registrar simulación</button>
    </form>}
  </>
}

type Props = {
  activo: boolean
  seleccion: SeleccionActiva | null
  selector: Omit<ComponentProps<typeof SelectorSerie>, 'idPrefix'>
}

export default function Simulaciones({ activo, seleccion, selector }: Props) {
  const [estado, setEstado] = useState<Envio>({ tipo: 'borrador' })
  const [confirmada, setConfirmada] = useState<ReturnType<typeof leerConfirmacion> | null>(null)
  const [pedirDescarte, setPedirDescarte] = useState(false)
  const historial = useRef<ControlHistorial>(null)
  const ocupada = useRef(false)

  async function registrar(intencion: Intencion, reintento = false) {
    if (ocupada.current) return
    ocupada.current = true
    setPedirDescarte(false)
    setEstado({ tipo: 'enviando', intencion,
      mensaje: reintento ? 'Reenviando la misma solicitud…' : 'Enviando la solicitud de registro…' })
    try {
      const resultado = await enviar(intencion)
      if (resultado.tipo === 'confirmado') {
        setConfirmada(resultado.registro)
        setEstado({ tipo: 'borrador' })
        historial.current?.refrescar()
      } else if (resultado.tipo === 'desconocido') {
        setEstado({ tipo: 'desconocido', intencion, mensaje: resultado.mensaje })
      } else {
        setEstado({
          tipo: resultado.tipo, intencion, mensaje: resultado.mensaje, http: resultado.http,
          reintento, conservar: reintento || resultado.persistenciaSinConfirmar,
        })
      }
    } finally {
      ocupada.current = false
    }
  }

  // La intención permanece en el componente padre al salir de la pestaña.
  // Se desmontan únicamente los paneles que hacen consultas GET.
  if (!activo) return null
  const segura = seleccion && Number.isSafeInteger(seleccion.conjuntoId) && seleccion.conjuntoId > 0 &&
    Number.isSafeInteger(seleccion.serie.id) && seleccion.serie.id > 0 &&
    Number.isSafeInteger(seleccion.serie.store_nbr) && seleccion.serie.store_nbr > 0
  const contexto = segura && seleccion ? claveContexto(seleccion) : null
  const rechazo = estado.tipo === 'error' || estado.tipo === 'conflicto' ? estado : null
  const pendiente = estado.tipo === 'desconocido' || rechazo?.conservar === true
  const bloqueado = estado.tipo === 'enviando' || pendiente
  const origen = estado.tipo === 'borrador' ? null : estado.intencion
  const visible = confirmada?.contexto === contexto ? confirmada : null

  return <div className="simulaciones-paneles">
    <section className="simulaciones-card" aria-labelledby="simulaciones-creacion">
      <h2 id="simulaciones-creacion">Crear nueva simulación</h2>
      <p>Los parámetros son hipotéticos. El dataset no contiene inventario observado y el registro no genera órdenes de compra.</p>
      <SelectorSerie {...selector} idPrefix="simulaciones-creacion" />
      {seleccion && !segura && <p role="alert">La selección contiene un identificador fuera del rango seguro del navegador.</p>}
      {!seleccion && <p role="status">Seleccione una tienda y una familia del conjunto activo para crear una simulación.</p>}
      <p id="simulaciones-parametros-ayuda" className="help-text">Ingrese cantidades no negativas con punto decimal. Use 0 para cero entero, 0.0 para cero decimal y -0.0 para cero negativo. Cada cantidad admite hasta 64 caracteres. El plazo admite enteros de 0 a 15 días.</p>
      {segura && seleccion && <FormularioSimulacion key={contexto} seleccion={seleccion}
        bloqueado={bloqueado} registrar={intencion => { void registrar(intencion) }} />}
      {origen && <p>Solicitud del pronóstico {origen.pronostico.pronostico} · conjunto {origen.conjunto} ·
        {' '}serie {origen.serie.id} · tienda {origen.serie.tienda} · {origen.serie.familia}.</p>}
      {estado.tipo === 'enviando' && <p role="status">{estado.mensaje}</p>}
      {rechazo && <p role="alert">
        {rechazo.tipo === 'conflicto' ? 'Conflicto de registro. ' : ''}
        {rechazo.reintento ? 'La API rechazó el reintento' : 'La API rechazó la solicitud'} (HTTP {rechazo.http}).
        {' '}{rechazo.mensaje}
      </p>}
      {pendiente && origen && <div className="simulaciones-pendiente">
        {estado.tipo === 'desconocido' ? <p role="alert">Confirmación desconocida. {estado.mensaje}
          {' '}Es posible que la ejecución ya se haya guardado.</p> :
          <p role="status">{rechazo?.reintento
            ? 'El resultado del envío anterior sigue sin confirmarse.'
            : 'El estado de la ejecución sigue sin confirmarse.'} La intención original se conserva.</p>}
        <div className="simulaciones-acciones">
          <button type="button" onClick={() => { void registrar(origen, true) }}>Reintentar la misma solicitud</button>
          <button type="button" onClick={() => setPedirDescarte(true)}>Descartar intención…</button>
        </div>
        {pedirDescarte && <div>
          <p role="alert">Descartar la intención no revierte un posible registro. Puede consultar el historial antes de iniciar otra solicitud.</p>
          <div className="simulaciones-acciones">
            <button type="button" onClick={() => { setEstado({ tipo: 'borrador' }); setPedirDescarte(false) }}>Confirmar descarte</button>
            <button type="button" onClick={() => setPedirDescarte(false)}>Conservar intención</button>
          </div>
        </div>}
      </div>}
      {confirmada && !visible && <div className="simulaciones-pendiente">
        <p role="status">Registro {confirmada.creado ? 'creado' : 'reutilizado'} para la tienda {confirmada.fila.serie.tienda} · {confirmada.fila.serie.familia}. Ejecución {confirmada.fila.id}.</p>
        <button type="button" onClick={() => { void historial.current?.abrir(confirmada.fila) }}>Consultar detalle registrado</button>
      </div>}
      {visible && <div className="simulaciones-pendiente">
        <p role="status">{visible.creado ? 'Ejecución creada' : 'Ejecución reutilizada'} · ID {visible.fila.id}.</p>
        <button type="button" onClick={() => { void historial.current?.abrir(visible.fila) }}>Consultar detalle registrado</button>
        <details><summary>Ver resultado confirmado</summary><Detalle datos={visible.datos} prefijo="simulaciones-confirmada" /></details>
      </div>}
    </section>
    <Historial control={historial} />
  </div>
}
