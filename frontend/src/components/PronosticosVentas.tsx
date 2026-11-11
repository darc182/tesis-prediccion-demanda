import { useCallback, useEffect, useId, useRef, useState, type FormEvent } from 'react'
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

type GrupoVariables = 'clasico' | 'ampliado'
const VARIABLES_CLASICAS = ['lag_1', 'lag_7', 'lag_14', 'media_7', 'dia_semana', 'mes'] as const
const VARIABLES_AMPLIADAS = [...VARIABLES_CLASICAS, 'media_14'] as const
type ParametrosSolicitados = Record<string, number | null>
type BorradorRF = { n_estimators: string; max_depth: string; min_samples_leaf: string }
type BorradorHGB = { max_iter: string; learning_rate: string; max_depth: string }
// Valores y límites de configuracion_experimentos.py; el backend sigue validando.
const INICIAL_RF: BorradorRF = { n_estimators: '100', max_depth: '', min_samples_leaf: '5' }
const INICIAL_HGB: BorradorHGB = { max_iter: '100', learning_rate: '0.05', max_depth: '' }

function numeroEnRango(texto: string, minimo: number, maximo: number, entero = true): number | null {
  const valor = Number(texto)
  return texto.trim() !== '' && Number.isFinite(valor) && valor >= minimo && valor <= maximo &&
    (!entero || Number.isInteger(valor)) ? valor : null
}

function resolverParametros(tecnica: Tecnica, rf: BorradorRF, hgb: BorradorHGB): ParametrosSolicitados | null {
  if (tecnica === 'estacional_semanal_recursiva') return {}
  const textoProfundidad = tecnica === 'random_forest' ? rf.max_depth : hgb.max_depth
  const profundidad = textoProfundidad.trim() === '' ? null : numeroEnRango(textoProfundidad, 1, 50)
  if (textoProfundidad.trim() !== '' && profundidad === null) return null
  let valores: ParametrosSolicitados
  let predeterminados: ParametrosSolicitados
  if (tecnica === 'random_forest') {
    const arboles = numeroEnRango(rf.n_estimators, 1, 500)
    const hoja = numeroEnRango(rf.min_samples_leaf, 1, 100)
    if (arboles === null || hoja === null) return null
    valores = { n_estimators: arboles, max_depth: profundidad, min_samples_leaf: hoja }
    predeterminados = { n_estimators: Number(INICIAL_RF.n_estimators), max_depth: null,
      min_samples_leaf: Number(INICIAL_RF.min_samples_leaf) }
  } else {
    const iteraciones = numeroEnRango(hgb.max_iter, 1, 500)
    const tasa = numeroEnRango(hgb.learning_rate, 0.001, 1, false)
    if (iteraciones === null || tasa === null) return null
    valores = { max_iter: iteraciones, learning_rate: tasa, max_depth: profundidad }
    predeterminados = { max_iter: Number(INICIAL_HGB.max_iter), learning_rate: Number(INICIAL_HGB.learning_rate), max_depth: null }
  }
  // Solo diferencias efectivas: 100.0 equivale a 100; vacío en profundidad es null.
  return Object.fromEntries(Object.entries(valores).filter(([nombre, valor]) => valor !== predeterminados[nombre]))
}

type SolicitudEnCurso = Seleccion & { corte: Corte } & (
  { tipo: 'individual'; tecnica: Tecnica; grupo: GrupoVariables | null; parametros: ParametrosSolicitados } |
  { tipo: 'bateria' }
)
type Metadatos = {
  id: number; experimento_id: number; tecnica: Tecnica; fecha_corte: Corte
  tipo: 'validacion'; horizonte: number; fecha_generacion: string
  inicio_entrenamiento: string; fin_entrenamiento: string
  duracion_segundos?: number | null
}
type Referencia = Pick<Metadatos, 'id' | 'experimento_id' | 'tecnica' | 'fecha_corte'>
type Fila = { paso: number; fecha_objetivo: string; venta_estimada: number }
type RespuestaDetalle = { conjunto_id: number; serie: SerieTemporal; pronostico: Metadatos; detalles: Fila[] }
type EstadoLista = { kind: 'loading' } | { kind: 'error'; message: string } | { kind: 'success'; filas: Metadatos[] }
type EstadoDetalle = { kind: 'idle' } | { kind: 'loading'; id: number } |
  { kind: 'error'; message: string } | { kind: 'success'; data: RespuestaDetalle }
type EstadoComparacion = { kind: 'idle' } | { kind: 'loading' } |
  { kind: 'error'; message: string } | { kind: 'success'; datos: RespuestaDetalle[] }
type ParametroGuardado = number | string | boolean | null
type ConfiguracionGuardada = {
  es_predeterminada: boolean
  configuracion: {
    tecnica: Tecnica; parametros: Record<string, ParametroGuardado>; variables: string[]
    horizonte: 15; version_contrato: 1 | 2; version_reglas: 1 | 2
  }
}
type EstadoConfiguracion = { kind: 'idle' } | { kind: 'loading'; id: number } |
  { kind: 'error'; id: number; message: string } |
  { kind: 'success'; pronostico: Metadatos; data: ConfiguracionGuardada }
const esEnteroGuardado = (valor: unknown) => typeof valor === 'number' && Number.isSafeInteger(valor)
const esRealGuardado = (valor: unknown) => typeof valor === 'number' && Number.isFinite(valor)
const esBooleanoGuardado = (valor: unknown) => typeof valor === 'boolean'
const esNuloGuardado = (valor: unknown) => valor === null
const esEnteroONuloGuardado = (valor: unknown) => esNuloGuardado(valor) || esEnteroGuardado(valor)
const esSemillaGuardada = (valor: unknown) => conteoValido(valor) && valor <= 0xffffffff

// Tipos de los parámetros efectivos publicados por F2. El backend valida los valores y sus límites.
const PARAMETROS_GUARDADOS: Record<Tecnica, Readonly<Record<string, (valor: unknown) => boolean>>> = {
  estacional_semanal_recursiva: { periodo_estacional_dias: esEnteroGuardado },
  random_forest: {
    bootstrap: esBooleanoGuardado,
    ccp_alpha: esRealGuardado,
    criterion: valor => valor === 'squared_error',
    max_depth: esEnteroONuloGuardado,
    max_features: valor => esRealGuardado(valor) || valor === 'sqrt' || valor === 'log2',
    max_leaf_nodes: esEnteroONuloGuardado,
    max_samples: esNuloGuardado,
    min_impurity_decrease: esRealGuardado,
    min_samples_leaf: esEnteroGuardado,
    min_samples_split: esEnteroGuardado,
    min_weight_fraction_leaf: esRealGuardado,
    monotonic_cst: esNuloGuardado,
    n_estimators: esEnteroGuardado,
    n_jobs: esEnteroGuardado,
    oob_score: esBooleanoGuardado,
    random_state: esSemillaGuardada,
    verbose: esEnteroGuardado,
    warm_start: esBooleanoGuardado,
  },
  hist_gradient_boosting: {
    categorical_features: valor => valor === 'from_dtype',
    early_stopping: esBooleanoGuardado,
    interaction_cst: esNuloGuardado,
    l2_regularization: esRealGuardado,
    learning_rate: esRealGuardado,
    loss: valor => valor === 'squared_error',
    max_bins: esEnteroGuardado,
    max_depth: esEnteroONuloGuardado,
    max_features: esRealGuardado,
    max_iter: esEnteroGuardado,
    max_leaf_nodes: esEnteroGuardado,
    min_samples_leaf: esEnteroGuardado,
    monotonic_cst: esNuloGuardado,
    n_iter_no_change: esEnteroGuardado,
    quantile: esNuloGuardado,
    random_state: esSemillaGuardada,
    scoring: valor => valor === 'loss',
    tol: esRealGuardado,
    validation_fraction: esRealGuardado,
    verbose: esEnteroGuardado,
    warm_start: esBooleanoGuardado,
  },
}
const ERRORES_CONFIGURACION: Readonly<Record<string, string>> = {
  pronostico_inexistente: 'El pronóstico no existe en el conjunto activo.',
  pronostico_corrupto: 'El pronóstico almacenado está incompleto o presenta datos inconsistentes.',
  sin_conjunto_activo: 'No existe un conjunto activo para esta operación.',
  base_no_disponible: 'La base de datos no está disponible o no tiene el esquema necesario.',
  error_interno: 'Ocurrió un error interno al consultar la configuración.',
}
type EstadoGeneracion = { kind: 'idle' } | { kind: 'success' | 'error'; message: string }
const ESTADOS_ELEMENTO = ['creado', 'reutilizado', 'fallido', 'no_ejecutado'] as const
type EstadoElemento = typeof ESTADOS_ELEMENTO[number]
type ErrorPublico = { codigo: string; mensaje: string }
type ElementoBateria = { tecnica: Tecnica; version_contrato: number } & (
  { estado: 'creado' | 'reutilizado'; duracion_segundos: number; experimento_id: number; pronostico_id: number } |
  { estado: 'fallido'; duracion_segundos: number; error: ErrorPublico } |
  { estado: 'no_ejecutado'; duracion_segundos: null }
)
type ResultadoBateria = {
  fecha_corte: Corte; estado: 'completa' | 'parcial' | 'interrumpida'
  resultados: ElementoBateria[]; resumen: Record<EstadoElemento, number>; interrupcion: ErrorPublico | null
}
type EstadoBateria = { kind: 'idle' } | { kind: 'success'; data: ResultadoBateria } |
  { kind: 'error'; message: string }
const ORDEN_BATERIA = TECNICAS.flatMap<{ tecnica: Tecnica; version: number }>(({ id }) => id === 'estacional_semanal_recursiva'
  ? [{ tecnica: id, version: 1 }]
  : [{ tecnica: id, version: 1 }, { tecnica: id, version: 2 }])
const ETIQUETAS_ELEMENTO: Record<EstadoElemento, string> = {
  creado: 'Creado', reutilizado: 'Reutilizado', fallido: 'Fallido', no_ejecutado: 'No ejecutado',
}
// Los mensajes se seleccionan por código público, sin presentar texto arbitrario de la respuesta.
const ERRORES_BATERIA: Readonly<Record<string, string>> = {
  entrada_bateria_invalida: 'Revise el conjunto y la fecha de corte.',
  conjunto_invalido: 'El identificador de conjunto no es válido.',
  serie_invalida: 'El identificador de serie no es válido.',
  corte_invalido: 'La fecha de corte no es válida.',
  corte_no_autorizado: 'El corte de validación no está autorizado.',
  prueba_reservada: 'La prueba final permanece reservada.',
  sin_conjunto_activo: 'No existe un conjunto activo.',
  conjunto_inactivo: 'El conjunto solicitado no es el activo. Actualice la selección.',
  serie_inexistente: 'La serie no existe en el conjunto activo.',
  serie_inactiva: 'La serie no pertenece al conjunto activo.',
  sin_historial: 'La serie no tiene observaciones hasta el corte.',
  historial_invalido: 'Las observaciones históricas presentan datos inconsistentes.',
  historial_cambiado: 'El conjunto, la serie o el histórico cambiaron.',
  base_no_disponible: 'La base de datos no está disponible o no tiene el esquema necesario.',
  error_calculo: 'El histórico no permite calcular este pronóstico.',
  resultado_invalido: 'El registro no acreditó un resultado válido para la configuración solicitada.',
  conflicto: 'Existe un pronóstico conflictivo. No se sobrescribió.',
  error_persistencia: 'No se pudo completar el registro de este pronóstico.',
  error_interno: 'No se pudo completar la ejecución de la batería.',
}
type Disponibilidad = {
  conjunto_id: number; serie: SerieTemporal; fecha_corte: Corte; grupo: GrupoVariables
  version_contrato: number; version_reglas: number; variables: string[]
  observaciones_historicas: number; filas_aptas: number; filas_descartadas: number
  descartes_por_variable: Record<string, number>
}
type EstadoDisponibilidad = { kind: 'idle' } | { kind: 'loading' } |
  { kind: 'error'; message: string } | { kind: 'success'; data: Disponibilidad }
const DIA = 86_400_000
// Formato de presentación; no se redondean ni reemplazan valores en el estado.
const formatoVentas = new Intl.NumberFormat('es-EC', { maximumFractionDigits: 3 })
const formatoCientifico = new Intl.NumberFormat('es-EC', { notation: 'scientific', maximumFractionDigits: 3 })
const formatoGeneracion = new Intl.DateTimeFormat('es-EC', { dateStyle: 'medium', timeStyle: 'short' })

function presentarVenta(valor: number): string {
  // Un positivo que se redondearía a cero conserva su orden de magnitud visible.
  return valor !== 0 && (valor < 0.001 || valor >= 1_000_000_000)
    ? formatoCientifico.format(valor) : formatoVentas.format(valor)
}

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
function conteoValido(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
}
function mensajeBateria(codigo: string): string | undefined {
  return Object.hasOwn(ERRORES_BATERIA, codigo) ? ERRORES_BATERIA[codigo] : undefined
}
function esErrorPublico(value: unknown): value is ErrorPublico {
  return registro(value) && typeof value.codigo === 'string' && typeof value.mensaje === 'string' &&
    value.mensaje.trim().length > 0 && mensajeBateria(value.codigo) !== undefined
}
function duracionValida(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
}
function esResultadoBateria(value: unknown, seleccion: Seleccion, corte: Corte): value is ResultadoBateria {
  if (!registro(value) || value.conjunto_id !== seleccion.conjuntoId || value.serie_id !== seleccion.serie.id ||
      value.fecha_corte !== corte || value.horizonte !== 15 || !duracionValida(value.duracion_segundos) ||
      !Array.isArray(value.resultados) || value.resultados.length !== ORDEN_BATERIA.length ||
      !registro(value.resumen)) return false

  const conteos = { creado: 0, reutilizado: 0, fallido: 0, no_ejecutado: 0 }
  const experimentos = new Set<number>()
  const pronosticos = new Set<number>()
  const identidades = new Set<string>()
  let debenQuedarPendientes = false
  for (const [indice, item] of value.resultados.entries()) {
    const { tecnica, version } = ORDEN_BATERIA[indice]
    const variables = tecnica === 'estacional_semanal_recursiva' ? [] :
      version === 1 ? VARIABLES_CLASICAS : VARIABLES_AMPLIADAS
    const columnasValidas = (columnas: unknown) => Array.isArray(columnas) && columnas.length === variables.length &&
      columnas.every((nombre, i) => nombre === variables[i])
    if (!registro(item) || item.orden !== indice + 1 || item.conjunto_id !== seleccion.conjuntoId ||
        item.serie_id !== seleccion.serie.id || item.fecha_corte !== corte || item.horizonte !== 15 ||
        item.tecnica !== tecnica || item.version_contrato !== version || item.version_reglas !== version ||
        !columnasValidas(item.variables) || typeof item.identidad_predictiva !== 'string' ||
        !/^[0-9a-f]{64}$/.test(item.identidad_predictiva) || identidades.has(item.identidad_predictiva)) return false
    identidades.add(item.identidad_predictiva)

    const config = item.configuracion
    if (!registro(config) || config.tecnica !== tecnica || config.horizonte !== 15 ||
        config.version_contrato !== version || config.version_reglas !== version || !columnasValidas(config.variables) ||
        !registro(config.parametros) || Object.keys(config.parametros).length === 0 ||
        !Object.values(config.parametros).every(p => p === null || typeof p === 'boolean' || typeof p === 'string' ||
          (typeof p === 'number' && Number.isFinite(p)))) return false

    if (debenQuedarPendientes && item.estado !== 'no_ejecutado') return false
    if (item.estado === 'creado' || item.estado === 'reutilizado') {
      if (!idValido(item.experimento_id) || !idValido(item.pronostico_id) || !duracionValida(item.duracion_segundos) ||
          experimentos.has(item.experimento_id) || pronosticos.has(item.pronostico_id) || Object.hasOwn(item, 'error')) return false
      experimentos.add(item.experimento_id)
      pronosticos.add(item.pronostico_id)
      conteos[item.estado]++
      continue
    }
    if (Object.hasOwn(item, 'experimento_id') || Object.hasOwn(item, 'pronostico_id')) return false
    if (item.estado === 'fallido') {
      if (!esErrorPublico(item.error) || !duracionValida(item.duracion_segundos)) return false
      debenQuedarPendientes = item.error.codigo !== 'error_calculo'
      conteos.fallido++
    } else if (item.estado === 'no_ejecutado') {
      if (item.duracion_segundos !== null || Object.hasOwn(item, 'error')) return false
      debenQuedarPendientes = true
      conteos.no_ejecutado++
    } else return false
  }

  const resumen = value.resumen
  if (Object.keys(resumen).length !== ESTADOS_ELEMENTO.length ||
      !ESTADOS_ELEMENTO.every(estado => resumen[estado] === conteos[estado])) return false
  if (value.estado === 'interrumpida') return value.contexto_verificado === false &&
    esErrorPublico(value.interrupcion) && value.interrupcion.codigo !== 'error_calculo'
  return value.contexto_verificado === true && value.interrupcion === null && !debenQuedarPendientes &&
    (conteos.fallido === 0 ? value.estado === 'completa' : value.estado === 'parcial')
}
function esDisponibilidad(value: unknown, seleccion: Seleccion, corte: Corte, grupo: GrupoVariables): value is Disponibilidad {
  const columnas = grupo === 'clasico' ? VARIABLES_CLASICAS : VARIABLES_AMPLIADAS
  const version = grupo === 'clasico' ? 1 : 2
  if (!registro(value) || !mismaSeleccion(value, seleccion) || value.fecha_corte !== corte || value.grupo !== grupo ||
      value.version_contrato !== version || value.version_reglas !== version ||
      !Array.isArray(value.variables) || value.variables.length !== columnas.length ||
      !value.variables.every((nombre, indice) => nombre === columnas[indice]) ||
      !conteoValido(value.observaciones_historicas) || !conteoValido(value.filas_aptas) ||
      !conteoValido(value.filas_descartadas) ||
      !Number.isSafeInteger(value.filas_aptas + value.filas_descartadas) ||
      value.filas_aptas + value.filas_descartadas !== value.observaciones_historicas ||
      !registro(value.descartes_por_variable)) return false
  const descartes = value.descartes_por_variable
  const filasDescartadas = value.filas_descartadas
  const motivos = Object.values(descartes)
  if (motivos.length !== columnas.length || !columnas.every(nombre => Object.hasOwn(descartes, nombre)) ||
      !motivos.every(conteoValido)) return false
  // Cada motivo afecta a filas descartadas; el calendario siempre está disponible.
  // La suma solo valida una cota de coherencia: nunca sustituye el total de la API.
  return motivos.every(n => n <= filasDescartadas) && descartes.dia_semana === 0 && descartes.mes === 0 &&
    motivos.reduce((suma, n) => suma + n, 0) >= filasDescartadas
}
function esMetadato(value: unknown): value is Metadatos {
  if (!registro(value)) return false
  return idValido(value.id) && idValido(value.experimento_id) && esTecnica(value.tecnica) &&
    esCorte(value.fecha_corte) && value.tipo === 'validacion' && value.horizonte === 15 &&
    (value.duracion_segundos == null || duracionValida(value.duracion_segundos)) &&
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
function esConfiguracionGuardada(value: unknown, seleccion: Seleccion, esperado: Metadatos): value is ConfiguracionGuardada {
  if (!registro(value) || !mismaSeleccion(value, seleccion) || value.pronostico_id !== esperado.id ||
      value.experimento_id !== esperado.experimento_id || value.fecha_corte !== esperado.fecha_corte ||
      typeof value.es_predeterminada !== 'boolean' || !registro(value.configuracion)) return false

  const config = value.configuracion
  const base = esperado.tecnica === 'estacional_semanal_recursiva'
  const version = config.version_contrato
  if (config.tecnica !== esperado.tecnica || config.horizonte !== esperado.horizonte ||
      (version !== 1 && (version !== 2 || base)) || config.version_reglas !== version ||
      !Array.isArray(config.variables) || !registro(config.parametros)) return false
  const variables = base ? [] : version === 1 ? VARIABLES_CLASICAS : VARIABLES_AMPLIADAS
  const columnas = config.variables
  if (columnas.length !== variables.length || new Set(columnas).size !== variables.length ||
      !variables.every(nombre => columnas.includes(nombre)) ||
      (version === 2 && !columnas.every((nombre, i) => nombre === variables[i]))) return false

  const parametros = config.parametros
  const tipos = Object.entries(PARAMETROS_GUARDADOS[esperado.tecnica])
  return Object.keys(parametros).length === tipos.length &&
    tipos.every(([nombre, validar]) => Object.hasOwn(parametros, nombre) && validar(parametros[nombre]))
}
function nombreTecnica(tecnica: Tecnica): string {
  return TECNICAS.find(t => t.id === tecnica)?.nombre ?? tecnica
}

function GraficoPronostico({ detalles, serie, pronostico }: {
  detalles: Fila[]; serie: SerieTemporal; pronostico: Metadatos
}) {
  const id = useId()
  const ancho = 760
  const alto = 340
  const izquierda = 122
  const derecha = 48
  const arriba = 24
  const abajo = 66
  const anchoGrafico = ancho - izquierda - derecha
  const altoGrafico = alto - arriba - abajo
  // Este componente solo recibe los quince detalles que ya superaron esDetalle.
  const inicio = instanteFecha(detalles[0].fecha_objetivo) as number
  const fin = instanteFecha(detalles[14].fecha_objetivo) as number
  const maximoReal = detalles.reduce((maximo, fila) => Math.max(maximo, fila.venta_estimada), 0)
  const maximo = maximoReal > 0 ? maximoReal : 1
  const posicionX = (fecha: string) => izquierda +
    ((instanteFecha(fecha) as number) - inicio) / (fin - inicio) * anchoGrafico
  // Dividir antes de multiplicar evita desbordamientos con valores grandes.
  // Con máximo positivo tampoco se resta un mínimo igual (casos constantes).
  const posicionY = (valor: number) => arriba + (1 - valor / maximo) * altoGrafico
  const marcasY = [...new Set([0, 0.25, 0.5, 0.75, 1].map(fraccion => maximo * fraccion))]
  const marcasX = [0, 3, 7, 10, 14].map(indice => detalles[indice])
  const puntos = detalles.map(fila => `${posicionX(fila.fecha_objetivo)},${posicionY(fila.venta_estimada)}`).join(' ')

  return <figure className="forecast-chart">
    <figcaption className="forecast-chart-caption">
      <strong>Ventas diarias estimadas</strong>
      <span>Tienda {serie.store_nbr} · {serie.family} · {nombreTecnica(pronostico.tecnica)}</span>
      <span>Corte {pronostico.fecha_corte} · Periodo {detalles[0].fecha_objetivo} a {detalles[14].fecha_objetivo}</span>
    </figcaption>
    <div className="forecast-chart-scroll" tabIndex={0} role="region" aria-label="Gráfico de las quince ventas estimadas, desplazable horizontalmente">
      <svg viewBox={`0 0 ${ancho} ${alto}`} role="img" aria-labelledby={`${id}-title ${id}-desc`}>
        <title id={`${id}-title`}>Quince ventas diarias estimadas de la tienda {serie.store_nbr}, {serie.family}</title>
        <desc id={`${id}-desc`}>
          Eje horizontal: fecha objetivo. Eje vertical lineal desde cero: ventas estimadas.
          La línea conecta los quince valores recibidos. Los ceros usan puntos huecos y los positivos rellenos.
          La tabla inferior ofrece los mismos datos y permite consultar cada valor completo.
        </desc>
        {marcasY.map(valor => <g key={valor}>
          <line x1={izquierda} x2={ancho - derecha} y1={posicionY(valor)} y2={posicionY(valor)} className="forecast-chart-grid" />
          <text x={izquierda - 10} y={posicionY(valor) + 4} textAnchor="end" className="forecast-chart-label">{presentarVenta(valor)}</text>
        </g>)}
        <line x1={izquierda} x2={izquierda} y1={arriba} y2={alto - abajo} className="forecast-chart-axis" />
        <line x1={izquierda} x2={ancho - derecha} y1={alto - abajo} y2={alto - abajo} className="forecast-chart-axis" />
        {marcasX.map(fila => <text key={fila.paso} x={posicionX(fila.fecha_objetivo)} y={alto - abajo + 23}
          textAnchor="middle" className="forecast-chart-label">{fila.fecha_objetivo}</text>)}
        <text x={izquierda + anchoGrafico / 2} y={alto - 7} textAnchor="middle" className="forecast-chart-axis-title">Fecha objetivo</text>
        <text transform={`translate(18 ${arriba + altoGrafico / 2}) rotate(-90)`} textAnchor="middle" className="forecast-chart-axis-title">Ventas estimadas</text>
        <polyline points={puntos} className="forecast-chart-line" />
        {detalles.map(fila => <circle key={fila.paso} cx={posicionX(fila.fecha_objetivo)} cy={posicionY(fila.venta_estimada)} r="4"
          className={`forecast-chart-point${fila.venta_estimada === 0 ? ' forecast-chart-point--zero' : ''}`}>
          <title>{fila.fecha_objetivo}: valor completo {String(fila.venta_estimada)}</title>
        </circle>)}
      </svg>
    </div>
    <p className="forecast-chart-note">Puntos huecos: estimación igual a cero. Puntos rellenos: valor positivo. Estimaciones basadas en ventas históricas.</p>
  </figure>
}

function GraficoComparacion({ datos }: { datos: RespuestaDetalle[] }) {
  const id = useId()
  const ancho = 1100
  const alto = 420
  const izquierda = 110
  const derecha = 35
  const arriba = 25
  const abajo = 125
  const maximoReal = datos.reduce((maximo, dato) =>
    dato.detalles.reduce((valor, fila) => Math.max(valor, fila.venta_estimada), maximo), 0)
  const maximo = maximoReal > 0 ? maximoReal : 1
  const x = (paso: number) => izquierda + (paso - 1) / 14 * (ancho - izquierda - derecha)
  const y = (valor: number) => arriba + (1 - valor / maximo) * (alto - arriba - abajo)
  const marcas = [...new Set([0, 0.25, 0.5, 0.75, 1].map(fraccion => maximo * fraccion))]
  const trazos = ['', '9 5', '2 5']
  const nombresTrazo = ['Continuo', 'Guiones', 'Puntos']

  return <>
    <figure className="forecast-chart">
      <figcaption className="forecast-chart-caption">
        <strong>Pronósticos almacenados superpuestos</strong>
        <span>Tienda {datos[0].serie.store_nbr} · {datos[0].serie.family} · Corte {datos[0].pronostico.fecha_corte}</span>
      </figcaption>
      <ul className="forecast-comparison-legend" aria-label="Leyenda de pronósticos comparados">
        {datos.map((dato, indice) => <li key={dato.pronostico.id}>
          <svg viewBox="0 0 50 12" aria-hidden="true"><line x1="0" x2="50" y1="6" y2="6"
            className={`forecast-comparison-line forecast-comparison-line--${indice}`} strokeDasharray={trazos[indice]} /></svg>
          <span>{nombreTecnica(dato.pronostico.tecnica)} · Pronóstico {dato.pronostico.id} · Experimento {dato.pronostico.experimento_id}
            {' · '}{nombresTrazo[indice]}</span>
        </li>)}
      </ul>
      <div className="forecast-chart-scroll forecast-comparison-scroll" tabIndex={0} role="region"
        aria-label="Gráfico conjunto de quince fechas, desplazable horizontalmente">
        <svg viewBox={`0 0 ${ancho} ${alto}`} role="img" aria-labelledby={`${id}-title ${id}-desc`}>
          <title id={`${id}-title`}>Comparación visual de {datos.length} pronósticos almacenados</title>
          <desc id={`${id}-desc`}>Quince fechas objetivo y eje vertical compartido desde cero. Cada pronóstico se identifica
            mediante su técnica, ID y trazo en la leyenda. La tabla inferior presenta las mismas estimaciones.</desc>
          {marcas.map(valor => <g key={valor}>
            <line x1={izquierda} x2={ancho - derecha} y1={y(valor)} y2={y(valor)} className="forecast-chart-grid" />
            <text x={izquierda - 10} y={y(valor) + 4} textAnchor="end" className="forecast-chart-label">{presentarVenta(valor)}</text>
          </g>)}
          <line x1={izquierda} x2={izquierda} y1={arriba} y2={alto - abajo} className="forecast-chart-axis" />
          <line x1={izquierda} x2={ancho - derecha} y1={alto - abajo} y2={alto - abajo} className="forecast-chart-axis" />
          {datos[0].detalles.map(fila => <text key={fila.paso}
            transform={`translate(${x(fila.paso)} ${alto - abajo + 20}) rotate(-45)`} textAnchor="end"
            className="forecast-chart-label">{fila.fecha_objetivo}</text>)}
          <text x={izquierda + (ancho - izquierda - derecha) / 2} y={alto - 8} textAnchor="middle"
            className="forecast-chart-axis-title">Fecha objetivo</text>
          <text transform={`translate(18 ${arriba + (alto - arriba - abajo) / 2}) rotate(-90)`} textAnchor="middle"
            className="forecast-chart-axis-title">Ventas estimadas</text>
          {datos.map((dato, indice) => <polyline key={dato.pronostico.id}
            points={dato.detalles.map(fila => `${x(fila.paso)},${y(fila.venta_estimada)}`).join(' ')}
            className={`forecast-comparison-line forecast-comparison-line--${indice}`} strokeDasharray={trazos[indice]}>
            <title>{nombreTecnica(dato.pronostico.tecnica)} · Pronóstico {dato.pronostico.id}</title>
          </polyline>)}
        </svg>
      </div>
    </figure>
    <div className="forecast-table-wrap" tabIndex={0} role="region" aria-label="Quince fechas de los pronósticos comparados">
      <table>
        <caption>Estimaciones por fecha objetivo</caption>
        <thead><tr><th scope="col">Fecha objetivo</th>{datos.map(dato => <th scope="col" key={dato.pronostico.id}>
          {nombreTecnica(dato.pronostico.tecnica)}<br />Pronóstico {dato.pronostico.id}
        </th>)}</tr></thead>
        <tbody>{datos[0].detalles.map((fila, indice) => <tr key={fila.paso}>
          <th scope="row">{fila.fecha_objetivo}</th>{datos.map(dato => <td key={dato.pronostico.id}>
            {presentarVenta(dato.detalles[indice].venta_estimada)}
          </td>)}
        </tr>)}</tbody>
      </table>
    </div>
    <p className="help-text">Las curvas son estimaciones. La comparación visual, por sí sola, no permite establecer
      cuál técnica es más precisa.</p>
  </>
}

class ErrorSolicitud extends Error {}
async function leerRespuesta(response: Response, mensajePublico?: (codigo: string) => string | undefined): Promise<unknown> {
  let body: unknown
  try { body = await response.json() } catch {
    throw new ErrorSolicitud(`El servidor respondió con HTTP ${response.status}, pero no devolvió un JSON válido.`)
  }
  if (!response.ok) {
    const mensaje = registro(body) && registro(body.detail) && typeof body.detail.codigo === 'string' &&
      typeof body.detail.mensaje === 'string' && body.detail.mensaje.trim()
      ? (mensajePublico ? mensajePublico(body.detail.codigo) : body.detail.mensaje) : null
    throw new ErrorSolicitud(mensaje ? `${mensaje} (HTTP ${response.status}).` : `La solicitud respondió con HTTP ${response.status}.`)
  }
  return body
}
function mensajeError(error: unknown, alternativa: string): string {
  return error instanceof ErrorSolicitud ? error.message : alternativa
}

function ConfiguracionPronostico({ pronostico, data }: { pronostico: Metadatos; data: ConfiguracionGuardada }) {
  const config = data.configuracion
  return <section className="forecast-results forecast-configuration" aria-labelledby="forecast-configuration-heading">
    <h4 id="forecast-configuration-heading">Configuración del pronóstico {pronostico.id}</h4>
    <p role="status">Experimento {pronostico.experimento_id} · {nombreTecnica(config.tecnica)} · Corte {pronostico.fecha_corte}</p>
    <dl className="forecast-configuration-fields">
      <div><dt>Configuración predeterminada</dt><dd>{data.es_predeterminada ? 'Sí' : 'No'}</dd></div>
      <div><dt>Horizonte</dt><dd>{config.horizonte} días</dd></div>
      <div><dt>Versión del contrato</dt><dd>{config.version_contrato}</dd></div>
      <div><dt>Versión de reglas</dt><dd>{config.version_reglas}</dd></div>
      <div><dt>Inicio de entrenamiento</dt><dd>{pronostico.inicio_entrenamiento}</dd></div>
      <div><dt>Fin de entrenamiento</dt><dd>{pronostico.fin_entrenamiento}</dd></div>
      <div><dt>Fecha de generación</dt><dd>{formatoGeneracion.format(new Date(pronostico.fecha_generacion))}</dd></div>
      {pronostico.duracion_segundos != null && <div><dt>Duración</dt><dd>{presentarVenta(pronostico.duracion_segundos)} s</dd></div>}
    </dl>
    <p>Variables predictoras {config.variables.length ? <code>{config.variables.join(', ')}</code> : '· No aplica'}</p>
    <h5>Parámetros efectivos</h5>
    <dl className="forecast-configuration-fields">
      {Object.entries(config.parametros).map(([nombre, valor]) => <div key={nombre}>
        <dt>{nombre === 'random_state' && 'Semilla aleatoria '}<code>{nombre}</code></dt><dd><code>{String(valor)}</code></dd>
      </div>)}
    </dl>
  </section>
}

function ConsultaDisponibilidad({ seleccion, corte, grupo, pendiente }: {
  seleccion: Seleccion; corte: Corte; grupo: GrupoVariables; pendiente: boolean
}) {
  const [estado, setEstado] = useState<EstadoDisponibilidad>({ kind: 'idle' })
  const solicitud = useRef<AbortController | null>(null)
  useEffect(() => () => {
    solicitud.current?.abort()
    solicitud.current = null
  }, [])

  async function consultar() {
    // La referencia bloquea dobles clics incluso antes del siguiente render.
    if (pendiente || solicitud.current !== null) return
    const controller = new AbortController()
    solicitud.current = controller
    setEstado({ kind: 'loading' })
    const vigente = () => solicitud.current === controller && !controller.signal.aborted
    try {
      const params = new URLSearchParams({ fecha_corte: corte, grupo })
      const response = await fetch(`/api/pronosticos/active/series/${seleccion.serie.id}/filas-entrenamiento?${params}`, {
        signal: controller.signal, cache: 'no-store',
      })
      const body = await leerRespuesta(response)
      if (!vigente()) return
      if (response.status !== 200 || !esDisponibilidad(body, seleccion, corte, grupo)) throw new ErrorSolicitud(
        'La disponibilidad recibida no corresponde a la selección o contiene conteos inconsistentes.')
      setEstado({ kind: 'success', data: body })
    } catch (error) {
      if (vigente()) setEstado({ kind: 'error', message: mensajeError(error,
        'No se pudo conectar para consultar la disponibilidad de filas de entrenamiento.') })
    } finally {
      if (solicitud.current === controller) solicitud.current = null
    }
  }

  return <div aria-busy={estado.kind === 'loading'}>
    <p>Grupo {grupo === 'clasico' ? 'clásico — 6 variables' : 'ampliado — 7 variables'} · Corte {corte}.</p>
    <button id="forecast-availability-query" type="button" disabled={pendiente || estado.kind === 'loading'}
      onClick={() => { void consultar() }}>Consultar disponibilidad</button>
    {estado.kind === 'idle' && <p role="status">Consulte las filas disponibles para la serie, el corte y el grupo seleccionados.</p>}
    {estado.kind === 'loading' && <p role="status">Consultando disponibilidad de filas de entrenamiento…</p>}
    {estado.kind === 'error' && <p className="forecast-message forecast-message--error" role="alert">{estado.message}</p>}
    {estado.kind === 'success' && <>
      <p role="status">Disponibilidad consultada para el grupo {estado.data.grupo === 'clasico' ? 'clásico' : 'ampliado'} y corte {estado.data.fecha_corte}.</p>
      <dl className="forecast-availability-counts">
        <div><dt>Observaciones históricas registradas hasta el corte</dt><dd>{estado.data.observaciones_historicas}</dd></div>
        <div><dt>Filas aptas para preparar ejemplos de entrenamiento</dt><dd>{estado.data.filas_aptas}</dd></div>
        <div><dt>Filas descartadas por información temporal insuficiente</dt><dd>{estado.data.filas_descartadas}</dd></div>
      </dl>
      {estado.data.filas_aptas === 0 && <p className="forecast-message" role="status">
        No hay filas aptas para preparar ejemplos de entrenamiento con esta selección. Estos datos no permiten entrenar el regresor.
      </p>}
      <div className="forecast-table-wrap" tabIndex={0} role="region" aria-label="Descartes por variable predictora">
        <table>
          <caption>Motivos de descarte por variable predictora</caption>
          <thead><tr><th scope="col">Variable predictora</th><th scope="col">Filas</th></tr></thead>
          <tbody>{estado.data.variables.map(nombre => <tr key={nombre}>
            <th scope="row"><code>{nombre}</code></th><td>{estado.data.descartes_por_variable[nombre]}</td>
          </tr>)}</tbody>
        </table>
      </div>
    </>}
    <p className="help-text">Una misma fila puede tener varios motivos de descarte; sus conteos pueden solaparse y no se suman para obtener el total descartado.
      Las fechas sin registro no cuentan como observaciones descartadas. Las ventas cero observadas sí son observaciones.</p>
    <p className="help-text">Las filas aptas indican disponibilidad de variables; no demuestran que el historial sea estadísticamente suficiente.</p>
  </div>
}

type PanelProps = {
  seleccion: Seleccion; activo: boolean; pendiente: boolean; revision: number
  iniciar: (solicitud: SolicitudEnCurso) => boolean
  finalizar: (cambioSeleccion: boolean) => void
}

function PanelPronosticos({ seleccion, activo, pendiente, revision, iniciar, finalizar }: PanelProps) {
  const { conjuntoId, serie: { id: serieId, store_nbr: tienda, family: familia } } = seleccion
  const [tecnica, setTecnica] = useState<Tecnica>('estacional_semanal_recursiva')
  const [corte, setCorte] = useState<Corte>('2017-06-16')
  const [grupo, setGrupo] = useState<GrupoVariables>('clasico')
  const [rf, setRf] = useState<BorradorRF>(() => ({ ...INICIAL_RF }))
  const [hgb, setHgb] = useState<BorradorHGB>(() => ({ ...INICIAL_HGB }))
  const esLineaBase = tecnica === 'estacional_semanal_recursiva'
  const parametros = resolverParametros(tecnica, rf, hgb)
  const modificados = parametros !== null && Object.keys(parametros).length > 0
  const [lista, setLista] = useState<EstadoLista>({ kind: 'loading' })
  const [filtroTecnica, setFiltroTecnica] = useState<Tecnica | ''>('')
  const [filtroCorte, setFiltroCorte] = useState<Corte | ''>('')
  const filasFiltradas = lista.kind === 'success' ? lista.filas.filter(p =>
    (!filtroTecnica || p.tecnica === filtroTecnica) && (!filtroCorte || p.fecha_corte === filtroCorte)) : []
  const [detalle, setDetalle] = useState<EstadoDetalle>({ kind: 'idle' })
  const [seleccionComparacion, setSeleccionComparacion] = useState<Metadatos[]>([])
  const [comparacion, setComparacion] = useState<EstadoComparacion>({ kind: 'idle' })
  const [configuracionGuardada, setConfiguracionGuardada] = useState<EstadoConfiguracion>({ kind: 'idle' })
  const [generacion, setGeneracion] = useState<EstadoGeneracion>({ kind: 'idle' })
  const [bateria, setBateria] = useState<EstadoBateria>({ kind: 'idle' })
  const solicitudLista = useRef<AbortController | null>(null)
  const solicitudDetalle = useRef<AbortController | null>(null)
  const solicitudConfiguracion = useRef<AbortController | null>(null)
  const solicitudComparacion = useRef<AbortController | null>(null)
  const vivo = useRef(true)
  const ultimaRevision = useRef(-1)

  useEffect(() => {
    vivo.current = true
    return () => {
      vivo.current = false
      ultimaRevision.current = -1 // permite repetir el GET abortado por StrictMode
      solicitudLista.current?.abort()
      solicitudDetalle.current?.abort()
      solicitudConfiguracion.current?.abort()
      solicitudComparacion.current?.abort()
    }
  }, [])

  const invalidarComparacion = useCallback(() => {
    solicitudComparacion.current?.abort()
    solicitudComparacion.current = null
    setComparacion({ kind: 'idle' })
  }, [])

  const consultarLista = useCallback(async () => {
    invalidarComparacion()
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
      const filas = body.pronosticos
      setSeleccionComparacion(anterior => anterior.flatMap(p => {
        const vigente = filas.find(fila => fila.id === p.id && fila.experimento_id === p.experimento_id &&
          fila.tecnica === p.tecnica && fila.fecha_corte === p.fecha_corte)
        return vigente ? [vigente] : []
      }))
      setLista({ kind: 'success', filas })
    } catch (error) {
      if (vivo.current && !controller.signal.aborted) setLista({ kind: 'error', message:
        mensajeError(error, 'No se pudo conectar para consultar los pronósticos almacenados.') })
    } finally {
      if (solicitudLista.current === controller) solicitudLista.current = null
    }
  }, [conjuntoId, serieId, tienda, familia, invalidarComparacion])

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

  function cerrarConfiguracion() {
    solicitudConfiguracion.current?.abort()
    solicitudConfiguracion.current = null
    setConfiguracionGuardada({ kind: 'idle' })
  }

  async function consultarConfiguracion(esperado: Metadatos) {
    solicitudConfiguracion.current?.abort()
    const controller = new AbortController()
    solicitudConfiguracion.current = controller
    setConfiguracionGuardada({ kind: 'loading', id: esperado.id })
    const vigente = () => vivo.current && solicitudConfiguracion.current === controller && !controller.signal.aborted
    try {
      const response = await fetch(`/api/pronosticos/active/${esperado.id}/configuracion`, {
        signal: controller.signal, cache: 'no-store',
      })
      const body = await leerRespuesta(response, codigo =>
        Object.hasOwn(ERRORES_CONFIGURACION, codigo) ? ERRORES_CONFIGURACION[codigo] : undefined)
      if (!vigente()) return
      if (response.status !== 200 || !esConfiguracionGuardada(body, seleccion, esperado)) {
        throw new ErrorSolicitud('La configuración recibida no corresponde al pronóstico seleccionado o contiene datos incompatibles.')
      }
      setConfiguracionGuardada({ kind: 'success', pronostico: esperado, data: body })
    } catch (error) {
      if (vigente()) setConfiguracionGuardada({ kind: 'error', id: esperado.id, message: mensajeError(error,
        'No se pudo conectar para consultar la configuración guardada.') })
    } finally {
      if (solicitudConfiguracion.current === controller) solicitudConfiguracion.current = null
    }
  }

  function motivoComparacion(p: Metadatos, elegidos = seleccionComparacion): string | null {
    if (elegidos.some(item => item.id === p.id)) return null
    if (elegidos.length >= 3) return 'Quite un pronóstico antes de añadir un cuarto.'
    if (elegidos.length && elegidos[0].fecha_corte !== p.fecha_corte) {
      return `Seleccione pronósticos del corte ${elegidos[0].fecha_corte}.`
    }
    if (elegidos.some(item => item.experimento_id === p.experimento_id)) {
      return 'Seleccione un experimento diferente.'
    }
    return null
  }

  function cambiarSeleccionComparacion(p: Metadatos, agregar: boolean) {
    if (agregar && (motivoComparacion(p) || seleccionComparacion.some(item => item.id === p.id))) return
    invalidarComparacion()
    setSeleccionComparacion(anterior => {
      if (!agregar) return anterior.filter(item => item.id !== p.id)
      return motivoComparacion(p, anterior) || anterior.some(item => item.id === p.id) ? anterior : [...anterior, p]
    })
  }

  function filtrarSeleccionComparacion(tecnicaHistorial: Tecnica | '', corteHistorial: Corte | '') {
    invalidarComparacion()
    setSeleccionComparacion(anterior => anterior.filter(p =>
      (!tecnicaHistorial || p.tecnica === tecnicaHistorial) && (!corteHistorial || p.fecha_corte === corteHistorial)))
  }

  async function consultarComparacion() {
    if (lista.kind !== 'success' || seleccionComparacion.length < 2 || seleccionComparacion.length > 3 ||
        solicitudComparacion.current !== null) return
    const esperados = [...seleccionComparacion]
    const controller = new AbortController()
    solicitudComparacion.current = controller
    setComparacion({ kind: 'loading' })
    const vigente = () => vivo.current && solicitudComparacion.current === controller && !controller.signal.aborted
    try {
      const datos = await Promise.all(esperados.map(async esperado => {
        const response = await fetch(`/api/pronosticos/active/${esperado.id}`, {
          signal: controller.signal, cache: 'no-store',
        })
        const body = await leerRespuesta(response, codigo => codigo === 'error_interno'
          ? 'Ocurrió un error interno al consultar el pronóstico.'
          : Object.hasOwn(ERRORES_CONFIGURACION, codigo) ? ERRORES_CONFIGURACION[codigo] : undefined)
        if (response.status !== 200 || !esDetalle(body, seleccion, esperado)) {
          throw new ErrorSolicitud('Los datos recibidos no corresponden a los pronósticos seleccionados o contienen estimaciones inválidas.')
        }
        return body
      }))
      if (!vigente()) return
      if (!datos.every(dato => dato.pronostico.fecha_corte === datos[0].pronostico.fecha_corte &&
          dato.detalles.every((fila, indice) => fila.fecha_objetivo === datos[0].detalles[indice].fecha_objetivo))) {
        throw new ErrorSolicitud('Los pronósticos recibidos tienen cortes o fechas objetivo diferentes.')
      }
      setComparacion({ kind: 'success', datos })
    } catch (error) {
      if (vigente()) {
        controller.abort()
        setComparacion({ kind: 'error', message: 'No se puede confirmar la comparación completa. ' +
          mensajeError(error, 'No se pudo conectar para consultar los pronósticos. Puede intentarlo nuevamente.') })
      }
    } finally {
      if (solicitudComparacion.current === controller) solicitudComparacion.current = null
    }
  }

  async function generar(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!esTecnica(tecnica) || !esCorte(corte) || parametros === null || !event.currentTarget.reportValidity() ||
        !iniciar({ tipo: 'individual', conjuntoId, serie: seleccion.serie, tecnica, corte,
          grupo: esLineaBase ? null : grupo, parametros })) return
    const configuracion = !esLineaBase && (grupo === 'ampliado' || modificados) ? {
      tecnica, version_contrato: grupo === 'ampliado' ? 2 : 1, version_reglas: grupo === 'ampliado' ? 2 : 1,
      variables: grupo === 'ampliado' ? VARIABLES_AMPLIADAS : VARIABLES_CLASICAS,
      ...(modificados ? { parametros } : {}),
    } : null
    setGeneracion({ kind: 'idle' })
    try {
      // Solo la acción de formulario inicia un POST. No se aborta ni reintenta:
      // cerrar una vista no implica cancelar el trabajo que recibe el servidor.
      const response = await fetch(`/api/pronosticos/active/series/${serieId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tecnica, fecha_corte: corte, ...(configuracion ? { configuracion } : {}) }),
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

  async function generarBateria() {
    if (!activo || !esCorte(corte) || !idValido(conjuntoId) || !idValido(serieId) ||
        !iniciar({ tipo: 'bateria', ...seleccion, corte })) return
    setBateria({ kind: 'idle' })
    try {
      const response = await fetch(`/api/pronosticos/active/series/${serieId}/bateria`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conjunto_id: conjuntoId, fecha_corte: corte }),
      })
      const body = await leerRespuesta(response, mensajeBateria)
      if (!vivo.current) return
      if (response.status !== 200 || !esResultadoBateria(body, seleccion, corte)) {
        throw new ErrorSolicitud('La respuesta no corresponde a la solicitud o contiene resultados inconsistentes.')
      }
      setBateria({ kind: 'success', data: body })
    } catch (error) {
      if (vivo.current) setBateria({ kind: 'error', message: 'No se pudo confirmar el resultado completo de la batería. ' +
        mensajeError(error, 'Se perdió la conexión. El servidor puede seguir procesando la solicitud.') +
        ' Consulte el listado antes de volver a ejecutar.' })
    } finally {
      if (vivo.current) await consultarLista()
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
            onChange={event => {
              if (!pendiente && esTecnica(event.target.value)) {
                setTecnica(event.target.value)
                if (event.target.value === 'estacional_semanal_recursiva') setGrupo('clasico')
              }
            }}>
            {TECNICAS.map(t => <option key={t.id} value={t.id}>{t.nombre}</option>)}
          </select>
        </div>
        <div>
          <label htmlFor="forecast-cutoff">Fecha de corte</label>
          <select id="forecast-cutoff" value={corte} disabled={pendiente}
            onChange={event => {
              if (!pendiente && esCorte(event.target.value)) {
                setCorte(event.target.value)
                setBateria({ kind: 'idle' })
              }
            }}>
            {CORTES.map((c, i) => <option key={c} value={c}>{c} · Validación {i + 1}</option>)}
          </select>
        </div>
        <div className="forecast-variable-group">
          <label htmlFor="forecast-variable-group">Grupo de variables predictoras</label>
          <select id="forecast-variable-group" value={grupo} disabled={pendiente || esLineaBase}
            aria-describedby="forecast-variable-help"
            onChange={event => {
              if (!pendiente && !esLineaBase && (event.target.value === 'clasico' || event.target.value === 'ampliado')) {
                setGrupo(event.target.value)
              }
            }}>
            {esLineaBase ? <option value="clasico">No aplica a la línea base</option> : <>
              <option value="clasico">Clásico — 6 variables</option>
              <option value="ampliado">Ampliado — 7 variables</option>
            </>}
          </select>
        </div>
        <div id="forecast-variable-help" className="help-text forecast-variable-help">
          {esLineaBase ? <p>La línea base semanal no utiliza un grupo de variables predictoras configurable.</p> : <>
            <p>Variables del grupo seleccionado: <code>{(grupo === 'ampliado' ? VARIABLES_AMPLIADAS : VARIABLES_CLASICAS).join(', ')}</code>.</p>
            <p>El grupo ampliado añade <code>media_14</code>: emplea las catorce fechas naturales anteriores y requiere
              observaciones para esas fechas. Las fechas sin registro no se sustituyen por ventas cero.</p>
          </>}
        </div>
        {!esLineaBase && <fieldset className="forecast-parameters" disabled={pendiente}
          aria-describedby="forecast-parameters-help">
          <legend>Hiperparámetros de {nombreTecnica(tecnica)}</legend>
          <div className="forecast-parameter-fields">
            {tecnica === 'random_forest' ? <>
              <div>
                <label htmlFor="forecast-n-estimators">Número de árboles (n_estimators)</label>
                <input id="forecast-n-estimators" type="number" min="1" max="500" step="1" required
                  value={rf.n_estimators} aria-describedby="forecast-n-estimators-help"
                  aria-invalid={numeroEnRango(rf.n_estimators, 1, 500) === null}
                  onChange={event => { if (!pendiente) setRf({ ...rf, n_estimators: event.target.value }) }} />
                <p id="forecast-n-estimators-help" className="help-text">Entero de 1 a 500. Predeterminado: 100.</p>
              </div>
              <div>
                <label htmlFor="forecast-min-samples-leaf">Mínimo de muestras por hoja (min_samples_leaf)</label>
                <input id="forecast-min-samples-leaf" type="number" min="1" max="100" step="1" required
                  value={rf.min_samples_leaf} aria-describedby="forecast-min-samples-leaf-help"
                  aria-invalid={numeroEnRango(rf.min_samples_leaf, 1, 100) === null}
                  onChange={event => { if (!pendiente) setRf({ ...rf, min_samples_leaf: event.target.value }) }} />
                <p id="forecast-min-samples-leaf-help" className="help-text">Entero de 1 a 100. Predeterminado: 5.</p>
              </div>
            </> : <>
              <div>
                <label htmlFor="forecast-max-iter">Iteraciones (max_iter)</label>
                <input id="forecast-max-iter" type="number" min="1" max="500" step="1" required
                  value={hgb.max_iter} aria-describedby="forecast-max-iter-help"
                  aria-invalid={numeroEnRango(hgb.max_iter, 1, 500) === null}
                  onChange={event => { if (!pendiente) setHgb({ ...hgb, max_iter: event.target.value }) }} />
                <p id="forecast-max-iter-help" className="help-text">Entero de 1 a 500. Predeterminado: 100.</p>
              </div>
              <div>
                <label htmlFor="forecast-learning-rate">Tasa de aprendizaje (learning_rate)</label>
                <input id="forecast-learning-rate" type="number" min="0.001" max="1" step="any" required
                  value={hgb.learning_rate} aria-describedby="forecast-learning-rate-help"
                  aria-invalid={numeroEnRango(hgb.learning_rate, 0.001, 1, false) === null}
                  onChange={event => { if (!pendiente) setHgb({ ...hgb, learning_rate: event.target.value }) }} />
                <p id="forecast-learning-rate-help" className="help-text">Entre 0,001 y 1. Predeterminado: 0,05.</p>
              </div>
            </>}
            <div>
              <label htmlFor="forecast-max-depth">Profundidad máxima (max_depth)</label>
              <input id="forecast-max-depth" type="number" min="1" max="50" step="1" placeholder="Sin límite"
                value={tecnica === 'random_forest' ? rf.max_depth : hgb.max_depth}
                aria-describedby="forecast-max-depth-help"
                aria-invalid={(tecnica === 'random_forest' ? rf.max_depth : hgb.max_depth).trim() !== '' &&
                  numeroEnRango(tecnica === 'random_forest' ? rf.max_depth : hgb.max_depth, 1, 50) === null}
                onChange={event => {
                  if (!pendiente) {
                    if (tecnica === 'random_forest') setRf({ ...rf, max_depth: event.target.value })
                    else setHgb({ ...hgb, max_depth: event.target.value })
                  }
                }} />
              <p id="forecast-max-depth-help" className="help-text">Vacío: sin límite, el valor predeterminado. O entero de 1 a 50.</p>
            </div>
          </div>
          <p id="forecast-parameters-help" className="help-text">Los demás parámetros conservan sus valores predeterminados. Los valores se validan antes de generar.</p>
          <p className="forecast-parameter-state" role="status">{parametros === null
            ? 'Hay parámetros inválidos. Revisa los límites indicados antes de generar.'
            : modificados ? 'Configuración de hiperparámetros modificada.' : 'Hiperparámetros predeterminados.'}</p>
          <button type="button" onClick={() => {
            if (!pendiente) {
              if (tecnica === 'random_forest') setRf({ ...INICIAL_RF })
              else setHgb({ ...INICIAL_HGB })
            }
          }}>Restablecer parámetros de {nombreTecnica(tecnica)}</button>
        </fieldset>}
        <p className="help-text">Horizonte fijo: quince días naturales desde el día siguiente al corte.</p>
        <button type="submit" disabled={pendiente || parametros === null || !esTecnica(tecnica) || !esCorte(corte)}>Generar pronóstico</button>
      </form>
      {generacion.kind !== 'idle' && <p className={`forecast-message forecast-message--${generacion.kind}`}
        role={generacion.kind === 'error' ? 'alert' : 'status'}>{generacion.message}</p>}

      <section className="forecast-results forecast-battery" aria-labelledby="forecast-battery-heading">
        <h3 id="forecast-battery-heading">Batería de pronósticos</h3>
        <p>Serie seleccionada · Corte {corte} · Horizonte de quince días.</p>
        <p className="help-text">Usa cinco configuraciones fijas: línea base semanal, Random Forest clásico y ampliado,
          e HistGradientBoosting clásico y ampliado. Los ajustes del formulario se aplican al pronóstico individual.</p>
        <button type="button" disabled={pendiente || !activo || !idValido(conjuntoId) || !idValido(serieId)}
          onClick={() => { void generarBateria() }}>Ejecutar batería de cinco configuraciones</button>
        {bateria.kind === 'error' && <p className="forecast-message forecast-message--error" role="alert">{bateria.message}</p>}
        {bateria.kind === 'success' && <>
          <p className="forecast-message" role="status">{bateria.data.estado === 'completa'
            ? 'Batería completa. Los cinco intentos terminaron correctamente.'
            : bateria.data.estado === 'parcial'
              ? 'Batería parcial. Hubo fallos de cálculo y se ejecutaron los cinco intentos.'
              : 'Batería interrumpida durante la ejecución o la verificación de los resultados.'}
            {bateria.data.interrupcion && ` ${mensajeBateria(bateria.data.interrupcion.codigo)}`}</p>
          <dl className="forecast-availability-counts">
            {ESTADOS_ELEMENTO.map(estado => <div key={estado}>
              <dt>{ETIQUETAS_ELEMENTO[estado]}</dt><dd>{bateria.data.resumen[estado]}</dd>
            </div>)}
          </dl>
          <div className="forecast-table-wrap" tabIndex={0} role="region" aria-label="Resultados de la batería">
            <table>
              <caption>Configuraciones ejecutadas y previstas para el corte {bateria.data.fecha_corte}</caption>
              <thead><tr><th scope="col">Técnica</th><th scope="col">Grupo</th><th scope="col">Estado</th>
                <th scope="col">Duración</th><th scope="col">Detalle</th></tr></thead>
              <tbody>{bateria.data.resultados.map((item, indice) => {
                const grupoItem = item.tecnica === 'estacional_semanal_recursiva' ? 'Sin grupo configurable' :
                  item.version_contrato === 1 ? 'Clásico — 6 variables' : 'Ampliado — 7 variables'
                return <tr key={indice}>
                  <th scope="row">{nombreTecnica(item.tecnica)}</th><td>{grupoItem}</td>
                  <td>{ETIQUETAS_ELEMENTO[item.estado]}
                    {item.estado === 'fallido' && <p>{mensajeBateria(item.error.codigo)}</p>}</td>
                  <td>{item.duracion_segundos === null ? '—' : `${presentarVenta(item.duracion_segundos)} s`}</td>
                  <td>{(item.estado === 'creado' || item.estado === 'reutilizado') && <button type="button"
                    disabled={detalle.kind === 'loading' && detalle.id === item.pronostico_id}
                    aria-label={`Ver detalle de ${nombreTecnica(item.tecnica)}, ${grupoItem}`}
                    onClick={() => { void consultarDetalle({ id: item.pronostico_id, experimento_id: item.experimento_id,
                      tecnica: item.tecnica, fecha_corte: corte }) }}>Ver detalle</button>}</td>
                </tr>
              })}</tbody>
            </table>
          </div>
        </>}
      </section>

      <section className="forecast-results forecast-availability" aria-labelledby="forecast-availability-heading">
        <h3 id="forecast-availability-heading">Disponibilidad de filas de entrenamiento</h3>
        {esLineaBase ? <p>Este análisis de variables predictoras no aplica a la línea base semanal.</p>
          : activo && <ConsultaDisponibilidad
            key={JSON.stringify([conjuntoId, serieId, tienda, familia, corte, grupo])}
            seleccion={seleccion} corte={corte} grupo={grupo} pendiente={pendiente} />}
      </section>

      <section className="forecast-results" aria-labelledby="forecast-list-heading">
        <div className="forecast-list-header">
          <h3 id="forecast-list-heading">Pronósticos almacenados</h3>
          <button type="button" onClick={() => { void consultarLista() }} disabled={lista.kind === 'loading'}>Actualizar listado</button>
        </div>
        <div className="forecast-form">
          <div>
            <label htmlFor="forecast-history-technique">Filtrar historial por técnica</label>
            <select id="forecast-history-technique" value={filtroTecnica} onChange={event => {
              const valor = event.target.value
              if (valor === '' || esTecnica(valor)) {
                setFiltroTecnica(valor)
                cerrarConfiguracion()
                filtrarSeleccionComparacion(valor, filtroCorte)
              }
            }}>
              <option value="">Todas las técnicas</option>
              {TECNICAS.map(t => <option key={t.id} value={t.id}>{t.nombre}</option>)}
            </select>
          </div>
          <div>
            <label htmlFor="forecast-history-cutoff">Filtrar historial por corte</label>
            <select id="forecast-history-cutoff" value={filtroCorte} onChange={event => {
              const valor = event.target.value
              if (valor === '' || esCorte(valor)) {
                setFiltroCorte(valor)
                cerrarConfiguracion()
                filtrarSeleccionComparacion(filtroTecnica, valor)
              }
            }}>
              <option value="">Todos los cortes</option>
              {CORTES.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
        </div>
        {lista.kind === 'loading' && <p role="status">Cargando pronósticos almacenados…</p>}
        {lista.kind === 'error' && <p className="forecast-message forecast-message--error" role="alert">{lista.message}</p>}
        {lista.kind === 'success' && (lista.filas.length === 0
          ? <p role="status">No hay pronósticos almacenados para esta serie.</p>
          : filasFiltradas.length === 0 ? <p role="status">No hay pronósticos que coincidan con los filtros seleccionados.</p>
          : <div className="forecast-table-wrap" tabIndex={0} role="region" aria-label="Listado de pronósticos">
            <table>
              <caption>Resultados disponibles de la serie seleccionada</caption>
              <thead><tr><th scope="col">Técnica</th><th scope="col">Corte</th><th scope="col">Periodo (15 días)</th>
                <th scope="col">Generación</th><th scope="col">Acción</th><th scope="col">Comparación visual</th></tr></thead>
              <tbody>{filasFiltradas.map(p => <tr key={p.id}>
                <td>{nombreTecnica(p.tecnica)}<br /><small>Pronóstico {p.id} · Experimento {p.experimento_id}</small></td><td>{p.fecha_corte}</td>
                <td>{fechaFutura(p.fecha_corte, 1)} a {fechaFutura(p.fecha_corte, 15)}</td>
                <td>{formatoGeneracion.format(new Date(p.fecha_generacion))}</td>
                <td className="forecast-history-actions"><button type="button" disabled={detalle.kind === 'loading' && detalle.id === p.id}
                  aria-label={`Ver pronóstico de ${nombreTecnica(p.tecnica)} con corte ${p.fecha_corte}`}
                  onClick={() => { void consultarDetalle(p) }}>Ver pronóstico</button>
                  <button type="button" disabled={configuracionGuardada.kind === 'loading' && configuracionGuardada.id === p.id}
                    aria-label={`Ver configuración del pronóstico ${p.id}`}
                    onClick={() => { void consultarConfiguracion(p) }}>Ver configuración</button></td>
                <td className="forecast-comparison-choice">
                  <label><input type="checkbox" checked={seleccionComparacion.some(item => item.id === p.id)}
                    disabled={motivoComparacion(p) !== null}
                    aria-label={`Comparar pronóstico ${p.id} de ${nombreTecnica(p.tecnica)} con corte ${p.fecha_corte}`}
                    aria-describedby={motivoComparacion(p) ? `forecast-compare-reason-${p.id}` : undefined}
                    onChange={event => cambiarSeleccionComparacion(p, event.target.checked)} /> Seleccionar</label>
                  {motivoComparacion(p) && <p id={`forecast-compare-reason-${p.id}`} className="help-text">{motivoComparacion(p)}</p>}
                </td>
              </tr>)}</tbody>
            </table>
          </div>)}
        {configuracionGuardada.kind === 'loading' && <p role="status">Consultando la configuración del pronóstico {configuracionGuardada.id}…</p>}
        {configuracionGuardada.kind === 'error' && <p className="forecast-message forecast-message--error" role="alert">
          Pronóstico {configuracionGuardada.id} · {configuracionGuardada.message}</p>}
        {configuracionGuardada.kind === 'success' && <ConfiguracionPronostico
          pronostico={configuracionGuardada.pronostico} data={configuracionGuardada.data} />}
      </section>
      <section className="forecast-results" aria-labelledby="forecast-comparison-heading" aria-busy={comparacion.kind === 'loading'}>
        <h3 id="forecast-comparison-heading">Comparación visual de pronósticos almacenados</h3>
        <p>Seleccione dos o tres pronósticos de experimentos diferentes y del mismo corte en el historial.</p>
        <p role="status">Seleccionados {seleccionComparacion.length} de 3
          {seleccionComparacion.length > 0 && ` · Corte ${seleccionComparacion[0].fecha_corte} · Pronósticos ${seleccionComparacion.map(p => p.id).join(', ')}`}.</p>
        <button type="button" disabled={lista.kind !== 'success' || seleccionComparacion.length < 2 || comparacion.kind === 'loading'}
          onClick={() => { void consultarComparacion() }}>Comparar pronósticos seleccionados</button>
        {comparacion.kind === 'loading' && <p role="status">Consultando todos los pronósticos seleccionados…</p>}
        {comparacion.kind === 'error' && <p role="alert" className="forecast-message forecast-message--error">{comparacion.message}</p>}
        {comparacion.kind === 'success' && <GraficoComparacion datos={comparacion.datos} />}
      </section>
      {detalle.kind === 'loading' && <p className="forecast-message" role="status">Consultando detalle del pronóstico…</p>}
      {detalle.kind === 'error' && <p className="forecast-message forecast-message--error" role="alert">{detalle.message}</p>}
      {detalle.kind === 'success' && <section className="forecast-results" aria-labelledby="forecast-detail-heading">
        <h3 id="forecast-detail-heading">{nombreTecnica(detalle.data.pronostico.tecnica)}</h3>
        <p>Tienda {tienda} · {familia} · Corte {detalle.data.pronostico.fecha_corte}</p>
        <p>Periodo: {detalle.data.detalles[0].fecha_objetivo} a {detalle.data.detalles[14].fecha_objetivo}.</p>
        <p className="forecast-explanation">Estimaciones basadas en ventas históricas.</p>
        <GraficoPronostico detalles={detalle.data.detalles} serie={detalle.data.serie} pronostico={detalle.data.pronostico} />
        <p className="forecast-value-help">Se muestran hasta tres decimales o notación científica para magnitudes extremas. Abra «Valor completo» para consultar el número recibido.</p>
        <div className="forecast-table-wrap" tabIndex={0} role="region" aria-label="Quince estimaciones diarias">
          <table>
            <caption>Pronóstico diario de demanda</caption>
            <thead><tr><th scope="col">Paso</th><th scope="col">Fecha objetivo</th><th scope="col">Venta estimada</th></tr></thead>
            <tbody>{detalle.data.detalles.map(fila => <tr key={fila.paso}>
              <td>{fila.paso}</td><td>{fila.fecha_objetivo}</td><td>
                <span>{presentarVenta(fila.venta_estimada)}</span>
                {fila.venta_estimada !== 0 && <details className="forecast-value-details">
                  <summary aria-label={`Valor completo del paso ${fila.paso}`}>Valor completo</summary>
                  <span>{String(fila.venta_estimada)}</span>
                </details>}
              </td>
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
  const [generando, setGenerando] = useState<SolicitudEnCurso | null>(null)
  const [revision, setRevision] = useState(0)
  const bloqueo = useRef(false)
  const iniciar = useCallback((solicitud: SolicitudEnCurso) => {
    if (bloqueo.current) return false
    bloqueo.current = true
    setGenerando(solicitud)
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
      <p>{generando.tipo === 'bateria' ? 'Ejecutando batería. El entrenamiento puede continuar hasta recibir la respuesta del servidor.'
        : 'Generando pronóstico. El entrenamiento puede tardar algunos segundos.'}</p>
      <p>Solicitud para conjunto {generando.conjuntoId} · tienda {generando.serie.store_nbr} · {generando.serie.family}.</p>
      {generando.tipo === 'bateria' ? <>
        <p>Corte {generando.corte}.</p><progress aria-label="Esperando la respuesta de la batería" />
      </> : <>
        <p>{nombreTecnica(generando.tecnica)} · Corte {generando.corte} · {generando.grupo === null
          ? 'Sin grupo configurable' : generando.grupo === 'ampliado' ? 'Ampliado — 7 variables' : 'Clásico — 6 variables'}.</p>
        {generando.grupo !== null && <p>{Object.keys(generando.parametros).length === 0 ? 'Hiperparámetros predeterminados.'
          : `Parámetros solicitados: ${Object.entries(generando.parametros).map(([nombre, valor]) => `${nombre}=${valor}`).join(', ')}.`}</p>}
      </>}
    </div>}
    {seleccion === null
      ? <p role="status">Seleccione una tienda y una familia válidas para consultar o generar pronósticos.</p>
      : <PanelPronosticos key={`${seleccion.conjuntoId}:${seleccion.serie.id}`} seleccion={seleccion}
        activo={activo} pendiente={generando !== null} revision={revision} iniciar={iniciar} finalizar={finalizar} />}
  </section>
}

export default PronosticosVentas
