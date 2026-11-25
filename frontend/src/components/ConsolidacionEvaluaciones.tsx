import { useEffect, useId, useRef, useState } from 'react'
import type { SerieTemporal } from './SelectorSerie'

type Seleccion = { conjuntoId: number; serie: SerieTemporal }
type Props = { seleccion: Seleccion }
const CORTES = ['2017-06-16', '2017-07-01', '2017-07-16'] as const
type Corte = typeof CORTES[number]
const TECNICAS = {
  estacional_semanal_recursiva: 'Línea base estacional semanal',
  random_forest: 'Random Forest',
  hist_gradient_boosting: 'HistGradientBoosting',
} as const
type Tecnica = keyof typeof TECNICAS
type Opcion = { id: number; experimento_id: number; tecnica: Tecnica; fecha_corte: Corte; fecha_generacion: string }
type Grupo = (Opcion | null)[]
type Parametro = number | boolean | string | null
type Configuracion = {
  version_contrato: 1 | 2; tecnica: Tecnica; parametros: Record<string, Parametro>
  variables: string[]; horizonte: 15; version_reglas: 1 | 2
}
type Versiones = { python: string; pandas: string; scikit_learn?: string }
type Cobertura = 'completa' | 'parcial' | 'sin_observaciones'
type Detalle = { paso: number; fecha_objetivo: string; venta_estimada: number; venta_real: number | null; error_absoluto: number | null }
type Evaluacion = {
  fecha_corte: Corte; horizonte: 15; observaciones_evaluadas: number; fechas_sin_registro: number
  cobertura: number; definicion_cobertura: string; mae: number | null; wape: number | null; rmse: number | null
  detalles: Detalle[]; fechas_ausentes: string[]
}
type Individual = {
  conjunto_id: number; serie: SerieTemporal; pronostico_id: number; experimento_id: number; tecnica: Tecnica
  fecha_corte: Corte; horizonte: 15; estado_cobertura: Cobertura; alcance_metricas: string; evaluacion: Evaluacion
}
type Consolidado = {
  mae: number | null; wape: number | null; rmse: number | null
  suma_errores_absolutos: number; suma_ventas_reales: number; observaciones_evaluadas: number
  fechas_sin_registro: number; cobertura: number; definicion_cobertura: string
  estado_cobertura: Cobertura; alcance_metricas: string
  rmse_escalado: { escala_errores: number; suma_cuadrados_normalizados: number }
}
type Resultado = {
  tecnica: Tecnica; identidad_configuracion: string; configuracion: Configuracion; versiones_ejecucion: Versiones
  evaluaciones: Individual[]; consolidado: Consolidado
}
type Consolidacion = {
  conjunto_id: number; serie: SerieTemporal; cortes: Corte[]; horizonte_por_corte: 15; fechas_esperadas_total: 45
  observaciones_evaluadas_total: number; cobertura_total: number; estado_cobertura: Cobertura; alcance_metricas: string
  fechas_ausentes_por_corte: { fecha_corte: Corte; fechas_ausentes: string[] }[]; resultados: Resultado[]
}
type VistaPrevia = {
  conjunto_id: number; serie: SerieTemporal; pronostico_id: number; experimento_id: number; fecha_corte: Corte
  configuracion: Configuracion; es_predeterminada: boolean
}
type Lista = { kind: 'loading' } | { kind: 'error'; mensaje: string } | { kind: 'success'; opciones: Opcion[]; total: number }
type Consulta = { kind: 'idle' | 'loading' } | { kind: 'error'; mensaje: string; detalleTecnico?: string } |
  { kind: 'success'; datos: Consolidacion; tiposMaxFeatures: (TipoMaxFeatures | undefined)[] }
type Inspeccion = { kind: 'idle' } | { kind: 'loading'; opcion: Opcion } |
  { kind: 'error'; mensaje: string } | { kind: 'success'; datos: VistaPrevia }

type TipoMaxFeatures = 'entero' | 'real'
type ConfiguracionAnalizada = { datos: VistaPrevia; tipoMaxFeatures?: TipoMaxFeatures }
type ResultadoAnalisis = { kind: 'valida'; configuracion: ConfiguracionAnalizada } |
  { kind: 'indeterminada'; id: number; mensaje: string } | { kind: 'error'; id: number; mensaje: string }
type Analisis = { kind: 'idle' | 'cancelado' } | { kind: 'loading'; solicitud: AbortController } |
  { kind: 'success'; resumen: ReturnType<typeof resumirAnalisis> }

const CONCURRENCIA_ANALISIS = 3

const ERRORES: Readonly<Record<string, string>> = {
  entrada_invalida: 'Complete dos o tres grupos, con un pronóstico distinto por corte y sin repetir IDs.',
  pronostico_invalido: 'Uno de los identificadores no es válido. Actualice el listado y revise la selección.',
  tipo_no_evaluable: 'Solo pueden consolidarse pronósticos de validación.',
  corte_no_autorizado: 'Los grupos deben utilizar los tres cortes de validación autorizados.',
  pronostico_inexistente: 'Uno de los pronósticos ya no existe. Actualice el listado.',
  pronostico_inactivo: 'Uno de los pronósticos no pertenece al conjunto activo. Actualice la selección.',
  sin_conjunto_activo: 'No hay un conjunto de datos activo para esta consulta.',
  configuraciones_no_equivalentes: 'Los tres pronósticos de un grupo no comparten configuración predictiva. Inspeccione sus configuraciones y revise el grupo.',
  configuraciones_duplicadas: 'Dos grupos representan la misma configuración. Seleccione configuraciones diferentes.',
  versiones_incompatibles: 'No se pudo consolidar. Los parámetros pueden coincidir aunque las versiones del software sean distintas o no se puedan verificar. Si el período tiene varias opciones, edite la selección y pruebe otro candidato del mismo período. Se conservarán las demás elecciones.',
  pronosticos_no_comparables: 'Los pronósticos no se pueden comparar con las mismas ventas reales. Compruebe la tienda, la familia y los períodos; luego prepare otra selección.',
  datos_inconsistentes: 'Hay datos incompletos o inconsistentes. Actualice el listado y revise los pronósticos de la selección.',
  resultado_no_finito: 'Los datos no permitieron calcular un resultado numérico válido. Revise los pronósticos en Comparación por corte.',
  base_no_disponible: 'La base de datos no está disponible. Puede solicitar de nuevo la consulta más tarde.',
  error_interno: 'No se pudo completar la consulta. Intente consultarla nuevamente.',
}
const ERROR_GENERICO = 'No se pudo completar la consulta. Compruebe la conexión y vuelva a solicitarla cuando lo desee.'
const RESPUESTA_INVALIDA = 'La respuesta está incompleta, es incompatible o no corresponde a la selección. No se muestran resultados.'
class ErrorConsulta extends Error {
  readonly detalleTecnico?: string

  constructor(mensaje: string, detalleTecnico?: string) {
    super(mensaje)
    this.detalleTecnico = detalleTecnico
  }
}
const ALCANCES: Record<Cobertura, string> = {
  completa: 'Métricas sobre las 45 observaciones de los tres horizontes.',
  parcial: 'Métricas sobre observaciones disponibles; no describen las fechas sin registro.',
  sin_observaciones: 'Sin observaciones reales; no se puede evaluar el desempeño predictivo.',
}
const ALCANCES_INDIVIDUALES: Record<Cobertura, string> = {
  completa: 'Métricas sobre las quince observaciones del horizonte.',
  parcial: 'Métricas sobre observaciones disponibles.',
  sin_observaciones: 'Sin observaciones reales; métricas no definidas.',
}
const VARIABLES = ['lag_1', 'lag_7', 'lag_14', 'media_7', 'dia_semana', 'mes']
const VERSION = /^[0-9]+(?:\.[0-9]+)+(?:a[0-9]+|b[0-9]+|rc[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?(?:\+[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*)?$/
const registro = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)
const numero = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= 0
const entero = (v: unknown): v is number => numero(v) && Number.isSafeInteger(v)
const idValido = (v: unknown): v is number => entero(v) && v > 0
const tecnica = (v: unknown): v is Tecnica => typeof v === 'string' && Object.hasOwn(TECNICAS, v)
const corteValido = (v: unknown): v is Corte => CORTES.some(c => c === v)
const nulo = (v: unknown) => v === null
const enteroNulo = (v: unknown) => v === null || entero(v)
const booleano = (v: unknown) => typeof v === 'boolean'
const mismasFechas = (v: unknown, fechas: readonly string[]): v is string[] =>
  Array.isArray(v) && v.length === fechas.length && v.every((f, i) => f === fechas[i])
function claves(v: Record<string, unknown>, nombres: string): boolean {
  const campos = nombres.split(' ')
  return Object.keys(v).length === campos.length && campos.every(k => Object.hasOwn(v, k))
}
function fecha(v: unknown): v is string {
  if (typeof v !== 'string' || !/^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(v) || v < '0001-01-01') return false
  const tiempo = Date.parse(`${v}T00:00:00Z`)
  return Number.isFinite(tiempo) && new Date(tiempo).toISOString().slice(0, 10) === v
}
function generacion(v: unknown): v is string {
  if (typeof v !== 'string' || !fecha(v.slice(0, 10))) return false
  const hora = /^(?:[T ]([0-9]{2}):([0-9]{2})(?::([0-9]{2})(?:\.[0-9]+)?)?(?:Z|[+-]([0-9]{2}):([0-9]{2}))?)?$/.exec(v.slice(10))
  return hora !== null && hora.slice(1).every((p, i) => p === undefined || Number(p) < (i === 0 || i === 3 ? 24 : 60)) &&
    Number.isFinite(Date.parse(v))
}
function fechaObjetivo(corte: Corte, paso: number): string {
  return new Date(Date.parse(`${corte}T00:00:00Z`) + paso * 86_400_000).toISOString().slice(0, 10)
}
function contexto(v: Record<string, unknown>, seleccion: Seleccion): boolean {
  return v.conjunto_id === seleccion.conjuntoId && registro(v.serie) && claves(v.serie, 'id store_nbr family') &&
    v.serie.id === seleccion.serie.id && v.serie.store_nbr === seleccion.serie.store_nbr && v.serie.family === seleccion.serie.family
}
function mismaOpcion(a: Opcion, b: Opcion): boolean {
  return a.id === b.id && a.experimento_id === b.experimento_id && a.tecnica === b.tecnica && a.fecha_corte === b.fecha_corte && a.fecha_generacion === b.fecha_generacion
}
function leerLista(v: unknown, seleccion: Seleccion): { opciones: Opcion[]; total: number } | null {
  if (!registro(v) || !claves(v, 'conjunto_id serie pronosticos') || !contexto(v, seleccion) || !Array.isArray(v.pronosticos)) return null
  const ids = new Set<number>()
  const opciones: Opcion[] = []
  for (const p of v.pronosticos) {
    if (!registro(p) || !claves(p, 'id experimento_id tecnica fecha_corte tipo horizonte fecha_generacion inicio_entrenamiento fin_entrenamiento duracion_segundos') ||
        !idValido(p.id) || !idValido(p.experimento_id) || ids.has(p.id) || !tecnica(p.tecnica) || !fecha(p.fecha_corte) ||
        typeof p.tipo !== 'string' || !['validacion', 'prueba', 'futuro'].includes(p.tipo) || p.horizonte !== 15 || !generacion(p.fecha_generacion) ||
        !fecha(p.inicio_entrenamiento) || !fecha(p.fin_entrenamiento) || p.inicio_entrenamiento > p.fin_entrenamiento ||
        p.fin_entrenamiento > p.fecha_corte || !(p.duracion_segundos === null || numero(p.duracion_segundos))) return null
    ids.add(p.id)
    if (p.tipo === 'validacion' && corteValido(p.fecha_corte)) {
      opciones.push({ id: p.id, experimento_id: p.experimento_id, tecnica: p.tecnica, fecha_corte: p.fecha_corte, fecha_generacion: p.fecha_generacion })
    }
  }
  return { opciones: opciones.sort((a, b) => a.id - b.id), total: v.pronosticos.length }
}

// Superficie pública F2. La acreditación definitiva de identidad y versiones corresponde a E2-A.
const PARAMETROS: Record<Tecnica, Readonly<Record<string, (v: unknown) => boolean>>> = {
  estacional_semanal_recursiva: { periodo_estacional_dias: entero },
  random_forest: {
    bootstrap: booleano, ccp_alpha: numero, criterion: v => v === 'squared_error', max_depth: enteroNulo,
    max_features: v => numero(v) || v === 'sqrt' || v === 'log2', max_leaf_nodes: enteroNulo, max_samples: nulo,
    min_impurity_decrease: numero, min_samples_leaf: entero, min_samples_split: entero, min_weight_fraction_leaf: numero,
    monotonic_cst: nulo, n_estimators: entero, n_jobs: entero, oob_score: booleano, random_state: entero,
    verbose: entero, warm_start: booleano,
  },
  hist_gradient_boosting: {
    categorical_features: v => v === 'from_dtype', early_stopping: booleano, interaction_cst: nulo,
    l2_regularization: numero, learning_rate: numero, loss: v => v === 'squared_error', max_bins: entero,
    max_depth: enteroNulo, max_features: numero, max_iter: entero, max_leaf_nodes: entero, min_samples_leaf: entero,
    monotonic_cst: nulo, n_iter_no_change: entero, quantile: nulo, random_state: entero,
    scoring: v => v === 'loss', tol: numero, validation_fraction: numero, verbose: entero, warm_start: booleano,
  },
}
function esConfiguracion(v: unknown, esperada: Tecnica): v is Configuracion {
  if (!registro(v) || !claves(v, 'version_contrato tecnica parametros variables horizonte version_reglas') ||
      v.tecnica !== esperada || v.horizonte !== 15 || (v.version_contrato !== 1 && v.version_contrato !== 2) ||
      v.version_reglas !== v.version_contrato || !Array.isArray(v.variables) || !registro(v.parametros)) return false
  const base = esperada === 'estacional_semanal_recursiva'
  if (base && v.version_contrato !== 1) return false
  const variables = base ? [] : v.version_contrato === 1 ? VARIABLES : [...VARIABLES, 'media_14']
  const columnas = v.variables
  if (columnas.length !== variables.length || new Set(columnas).size !== variables.length ||
      !variables.every(nombre => columnas.includes(nombre)) ||
      (v.version_contrato === 2 && !mismasFechas(columnas, variables))) return false
  const parametros = v.parametros
  const tipos = Object.entries(PARAMETROS[esperada])
  return Object.keys(parametros).length === tipos.length && tipos.every(([k, validar]) => Object.hasOwn(parametros, k) && validar(parametros[k]))
}
function esVersiones(v: unknown, t: Tecnica): v is Versiones {
  return registro(v) && claves(v, t === 'estacional_semanal_recursiva' ? 'python pandas' : 'python pandas scikit_learn') &&
    Object.values(v).every(x => typeof x === 'string' && VERSION.test(x))
}
function esVistaPrevia(v: unknown, seleccion: Seleccion, opcion: Opcion): v is VistaPrevia {
  return registro(v) && claves(v, 'conjunto_id serie pronostico_id experimento_id fecha_corte configuracion es_predeterminada') &&
    contexto(v, seleccion) && v.pronostico_id === opcion.id && v.experimento_id === opcion.experimento_id &&
    v.fecha_corte === opcion.fecha_corte && typeof v.es_predeterminada === 'boolean' && esConfiguracion(v.configuracion, opcion.tecnica)
}
function estadoCobertura(n: number, total: number): Cobertura {
  return n === 0 ? 'sin_observaciones' : n === total ? 'completa' : 'parcial'
}
function metricasValidas(v: Record<string, unknown>, cantidad: number, realesPositivos: boolean): boolean {
  return cantidad === 0 ? v.mae === null && v.wape === null && v.rmse === null :
    numero(v.mae) && numero(v.rmse) && (realesPositivos ? numero(v.wape) : v.wape === null)
}
function esEvaluacion(v: unknown, corte: Corte): v is Evaluacion {
  if (!registro(v) || !claves(v, 'fecha_corte horizonte observaciones_evaluadas fechas_sin_registro cobertura definicion_cobertura mae wape rmse detalles fechas_ausentes') ||
      v.fecha_corte !== corte || v.horizonte !== 15 || v.definicion_cobertura !== 'observaciones_evaluadas / horizonte (15 fechas naturales)' ||
      !Array.isArray(v.detalles) || v.detalles.length !== 15) return false
  const ausentes: string[] = []
  let positivos = false
  for (const [i, d] of v.detalles.entries()) {
    if (!registro(d) || !claves(d, 'paso fecha_objetivo venta_estimada venta_real error_absoluto') ||
        d.paso !== i + 1 || d.fecha_objetivo !== fechaObjetivo(corte, i + 1) || !numero(d.venta_estimada)) return false
    if (d.venta_real === null) {
      if (d.error_absoluto !== null) return false
      ausentes.push(d.fecha_objetivo as string)
    } else {
      if (!numero(d.venta_real) || !numero(d.error_absoluto)) return false
      positivos ||= d.venta_real > 0
    }
  }
  const cantidad = 15 - ausentes.length
  return v.observaciones_evaluadas === cantidad && v.fechas_sin_registro === ausentes.length &&
    v.cobertura === cantidad / 15 && mismasFechas(v.fechas_ausentes, ausentes) && metricasValidas(v, cantidad, positivos)
}
function esIndividual(v: unknown, seleccion: Seleccion, opcion: Opcion): v is Individual {
  if (!registro(v) || !claves(v, 'conjunto_id serie pronostico_id experimento_id tecnica fecha_corte horizonte estado_cobertura alcance_metricas evaluacion') ||
      !contexto(v, seleccion) || v.pronostico_id !== opcion.id || v.experimento_id !== opcion.experimento_id ||
      v.tecnica !== opcion.tecnica || v.fecha_corte !== opcion.fecha_corte || v.horizonte !== 15 || !esEvaluacion(v.evaluacion, opcion.fecha_corte)) return false
  const estado = estadoCobertura(v.evaluacion.observaciones_evaluadas, 15)
  return v.estado_cobertura === estado && v.alcance_metricas === ALCANCES_INDIVIDUALES[estado]
}
function esConsolidado(v: unknown, cantidad: number, positivos: boolean): v is Consolidado {
  if (!registro(v) || !claves(v, 'mae wape rmse suma_errores_absolutos suma_ventas_reales observaciones_evaluadas fechas_sin_registro cobertura definicion_cobertura estado_cobertura alcance_metricas rmse_escalado') ||
      !numero(v.suma_errores_absolutos) || !numero(v.suma_ventas_reales) || v.observaciones_evaluadas !== cantidad ||
      v.fechas_sin_registro !== 45 - cantidad || v.cobertura !== cantidad / 45 ||
      v.definicion_cobertura !== 'observaciones_evaluadas / 45 fechas naturales esperadas' ||
      !metricasValidas(v, cantidad, positivos) || !registro(v.rmse_escalado) ||
      !claves(v.rmse_escalado, 'escala_errores suma_cuadrados_normalizados')) return false
  const { escala_errores: escala, suma_cuadrados_normalizados: cuadrados } = v.rmse_escalado
  if (!numero(escala) || !numero(cuadrados) || cuadrados > cantidad ||
      (escala === 0 ? cuadrados !== 0 : cuadrados < 1) || (positivos ? v.suma_ventas_reales === 0 : v.suma_ventas_reales !== 0)) return false
  if (cantidad === 0 && (v.suma_errores_absolutos !== 0 || escala !== 0 || cuadrados !== 0)) return false
  const estado = estadoCobertura(cantidad, 45)
  return v.estado_cobertura === estado && v.alcance_metricas === ALCANCES[estado]
}
function esConsolidacion(v: unknown, seleccion: Seleccion, grupos: Opcion[][]): v is Consolidacion {
  if (!registro(v) || !claves(v, 'conjunto_id serie cortes horizonte_por_corte fechas_esperadas_total observaciones_evaluadas_total cobertura_total estado_cobertura alcance_metricas fechas_ausentes_por_corte resultados') ||
      !contexto(v, seleccion) || !mismasFechas(v.cortes, CORTES) || v.horizonte_por_corte !== 15 || v.fechas_esperadas_total !== 45 ||
      !Array.isArray(v.resultados) || v.resultados.length !== grupos.length || ![2, 3].includes(grupos.length) ||
      !Array.isArray(v.fechas_ausentes_por_corte) || v.fechas_ausentes_por_corte.length !== 3) return false
  const identidades = new Set<string>()
  let referencia: Individual[] | null = null
  for (const [i, r] of v.resultados.entries()) {
    if (!registro(r) || !claves(r, 'tecnica identidad_configuracion configuracion versiones_ejecucion evaluaciones consolidado') ||
        !tecnica(r.tecnica) || typeof r.identidad_configuracion !== 'string' || !/^[0-9a-f]{64}$/.test(r.identidad_configuracion) ||
        identidades.has(r.identidad_configuracion) || !esConfiguracion(r.configuracion, r.tecnica) || !esVersiones(r.versiones_ejecucion, r.tecnica) ||
        !Array.isArray(r.evaluaciones) || r.evaluaciones.length !== 3 || grupos[i].length !== 3) return false
    identidades.add(r.identidad_configuracion)
    let cantidad = 0
    let positivos = false
    const individuales: Individual[] = []
    for (const [j, e] of r.evaluaciones.entries()) {
      if (!esIndividual(e, seleccion, grupos[i][j]) || e.fecha_corte !== CORTES[j] || e.tecnica !== r.tecnica) return false
      const detallesReferencia = referencia?.[j].evaluacion.detalles
      if (detallesReferencia && e.evaluacion.detalles.some((d, k) => d.venta_real !== detallesReferencia[k].venta_real)) return false
      const ausencias = v.fechas_ausentes_por_corte[j]
      if (!registro(ausencias) || !claves(ausencias, 'fecha_corte fechas_ausentes') || ausencias.fecha_corte !== CORTES[j] ||
          !mismasFechas(ausencias.fechas_ausentes, e.evaluacion.fechas_ausentes)) return false
      // Solo se reúnen conteos; no se suman errores, ventas ni métricas diarias.
      cantidad += e.evaluacion.observaciones_evaluadas
      positivos ||= e.evaluacion.detalles.some(d => d.venta_real !== null && d.venta_real > 0)
      individuales.push(e)
    }
    if (!esConsolidado(r.consolidado, cantidad, positivos) || v.observaciones_evaluadas_total !== cantidad || v.cobertura_total !== cantidad / 45) return false
    referencia = individuales
  }
  if (!entero(v.observaciones_evaluadas_total) || v.observaciones_evaluadas_total > 45) return false
  const estado = estadoCobertura(v.observaciones_evaluadas_total, 45)
  return v.estado_cobertura === estado && v.alcance_metricas === ALCANCES[estado]
}
function gruposCompletos(grupos: Grupo[], opciones: Opcion[]): grupos is Opcion[][] {
  return [2, 3].includes(grupos.length) && grupos.every(g => g.length === 3 && g.every((p, j) =>
    p !== null && p.fecha_corte === CORTES[j] && opciones.some(o => mismaOpcion(o, p)))) &&
    new Set(grupos.flat().map(p => p?.id)).size === grupos.length * 3
}
async function obtener(url: string, signal: AbortSignal,
  leerRespuesta: (respuesta: Response) => Promise<unknown> = respuesta => respuesta.json()): Promise<unknown> {
  let respuesta: Response
  try { respuesta = await fetch(url, { method: 'GET', cache: 'no-store', signal }) }
  catch { throw new ErrorConsulta(ERROR_GENERICO) }
  let body: unknown
  try { body = await leerRespuesta(respuesta) }
  catch { throw new ErrorConsulta(RESPUESTA_INVALIDA) }
  if (!respuesta.ok) {
    const codigo = registro(body) && registro(body.detail) ? body.detail.codigo : null
    const detalle = registro(body) && registro(body.detail) ? body.detail.mensaje : null
    const mensaje = typeof codigo === 'string' && Object.hasOwn(ERRORES, codigo) ? ERRORES[codigo] : ERROR_GENERICO
    if (codigo === 'versiones_incompatibles' && url.startsWith('/api/evaluaciones/consolidacion/')) {
      throw new ErrorConsulta(mensaje, typeof detalle === 'string' ? `${codigo} · ${detalle}` : codigo)
    }
    throw new ErrorConsulta(typeof detalle === 'string' && url.startsWith('/api/evaluaciones/consolidacion/') ?
      `${detalle} ${mensaje}` : mensaje)
  }
  return body
}
async function leerRespuestaConTipos(respuesta: Response, tipos: WeakMap<object, TipoMaxFeatures>): Promise<unknown> {
  const texto = await respuesta.text()
  // F2 normaliza el tipo de los demás parámetros. RF admite un conteo entero
  // o una fracción en este mismo campo; el valor JS 1 por sí solo es ambiguo.
  return JSON.parse(texto, function (this: object, clave: string, valor: unknown, contexto?: { source: string }) {
    if (clave === 'max_features' && typeof valor === 'number' && contexto) {
      tipos.set(this, /^-?[0-9]+$/.test(contexto.source) ? 'entero' : 'real')
    }
    return valor
  })
}

async function analizarConfiguracion(seleccion: Seleccion, opcion: Opcion, signal: AbortSignal): Promise<ResultadoAnalisis> {
  const tipos = new WeakMap<object, TipoMaxFeatures>()
  try {
    const body = await obtener(`/api/pronosticos/active/${opcion.id}/configuracion`, signal,
      respuesta => leerRespuestaConTipos(respuesta, tipos))
    if (!esVistaPrevia(body, seleccion, opcion)) throw new ErrorConsulta(RESPUESTA_INVALIDA)
    const config = body.configuracion
    let tipoMaxFeatures: TipoMaxFeatures | undefined
    if (config.tecnica === 'random_forest' && typeof config.parametros.max_features === 'number') {
      const valor = config.parametros.max_features
      if (valor < 0.01 || valor > config.variables.length) throw new ErrorConsulta(RESPUESTA_INVALIDA)
      tipoMaxFeatures = tipos.get(config.parametros) ?? (Number.isInteger(valor) ? undefined : 'real')
      if (!tipoMaxFeatures) return {
        kind: 'indeterminada', id: opcion.id,
        mensaje: 'Este navegador no puede comprobar el tipo numérico original de max_features.',
      }
      if (tipoMaxFeatures === 'entero' ? !Number.isSafeInteger(valor) || valor < 1 || valor > config.variables.length :
        valor < 0.01 || valor > 1) throw new ErrorConsulta(RESPUESTA_INVALIDA)
    }
    return { kind: 'valida', configuracion: { datos: body, tipoMaxFeatures } }
  } catch (error) {
    return { kind: 'error', id: opcion.id, mensaje: error instanceof ErrorConsulta ? error.message : RESPUESTA_INVALIDA }
  }
}

async function analizarDisponibles(seleccion: Seleccion, opciones: Opcion[], signal: AbortSignal) {
  const resultados: ResultadoAnalisis[] = []
  let siguiente = 0
  async function consultar() {
    while (!signal.aborted && siguiente < opciones.length) {
      const indice = siguiente++
      resultados[indice] = await analizarConfiguracion(seleccion, opciones[indice], signal)
    }
  }
  await Promise.all(Array.from({ length: Math.min(CONCURRENCIA_ANALISIS, opciones.length) }, consultar))
  return resultados
}

function mismaConfiguracionPublica(a: ConfiguracionAnalizada, b: ConfiguracionAnalizada): boolean {
  const x = a.datos.configuracion
  const y = b.datos.configuracion
  // Las claves de parámetros ya fueron verificadas. El orden del objeto es
  // irrelevante, pero el de las variables sí forma parte del contrato F2.
  return x.tecnica === y.tecnica && x.version_contrato === y.version_contrato &&
    x.version_reglas === y.version_reglas && x.horizonte === y.horizonte &&
    mismasFechas(x.variables, y.variables) && a.tipoMaxFeatures === b.tipoMaxFeatures &&
    Object.entries(x.parametros).every(([clave, valor]) => valor === y.parametros[clave])
}

type Disponibilidad = { referente: ConfiguracionAnalizada; cortes: Map<Corte, number[]> }
type Borrador = { configuracion: Disponibilidad; candidatos: Partial<Record<Corte, number>> }

function resumirAnalisis(resultados: ResultadoAnalisis[]) {
  const grupos: Disponibilidad[] = []
  const incidencias: Exclude<ResultadoAnalisis, { kind: 'valida' }>[] = []
  for (const resultado of resultados) {
    if (resultado.kind !== 'valida') {
      incidencias.push(resultado)
      continue
    }
    const actual = resultado.configuracion
    let grupo = grupos.find(g => mismaConfiguracionPublica(g.referente, actual))
    if (!grupo) {
      grupo = { referente: actual, cortes: new Map() }
      grupos.push(grupo)
    }
    const { fecha_corte: corte, pronostico_id: id } = actual.datos
    const ids = grupo.cortes.get(corte)
    if (ids) ids.push(id)
    else grupo.cortes.set(corte, [id])
  }
  return { grupos, incidencias }
}

const formato = new Intl.NumberFormat('es-EC', { maximumSignificantDigits: 6 })
const cientifico = new Intl.NumberFormat('es-EC', { notation: 'scientific', maximumSignificantDigits: 6 })
function presentarNumero(v: number): string {
  return v !== 0 && (v < 0.0001 || v >= 1_000_000_000) ? cientifico.format(v) : formato.format(v)
}
function Valor({ valor, ausente = 'No definido' }: { valor: number | null; ausente?: string }) {
  if (valor === null) return <span>{ausente}</span>
  const visible = presentarNumero(valor)
  return <><span title={String(valor)}>{visible}</span>{visible !== String(valor) && <details className="evaluation-value">
    <summary>Valor completo</summary><code>{String(valor)}</code>
  </details>}</>
}
function etiqueta(p: Opcion): string {
  return `${TECNICAS[p.tecnica]} · pronóstico ${p.id} · experimento ${p.experimento_id} · ${p.fecha_corte}`
}
function ConfiguracionPublica({ datos }: { datos: Configuracion }) {
  return <div className="consolidation-configuration">
    <dl><dt>Técnica</dt><dd>{TECNICAS[datos.tecnica]}</dd><dt>Contrato / reglas</dt><dd>{datos.version_contrato} / {datos.version_reglas}</dd>
      <dt>Horizonte por corte</dt><dd>{datos.horizonte} días</dd></dl>
    <p>Variables en el orden declarado: {datos.variables.length === 0 ? 'la línea base no utiliza variables predictoras.' :
      datos.version_contrato === 1 ? 'grupo clásico.' : 'grupo ampliado.'}</p>
    {datos.variables.length > 0 && <ol>{datos.variables.map(v => <li key={v}><code>{v}</code></li>)}</ol>}
    <div className="evaluation-scroll consolidation-scroll" role="region" tabIndex={0} aria-label="Parámetros efectivos desplazables">
      <table><caption>Parámetros efectivos completos</caption><thead><tr><th scope="col">Parámetro</th><th scope="col">Valor</th></tr></thead>
        <tbody>{Object.entries(datos.parametros).map(([k, v]) => <tr key={k}><th scope="row"><code>{k}</code></th>
          <td>{typeof v === 'number' ? <Valor valor={v} /> : <code>{String(v)}</code>}</td></tr>)}</tbody></table>
    </div>
  </div>
}
const PARAMETROS_VISIBLES: Record<Tecnica, Record<string, string>> = {
  estacional_semanal_recursiva: { periodo_estacional_dias: 'Ciclo en días' },
  random_forest: {
    n_estimators: 'Árboles', max_depth: 'Profundidad máxima', max_features: 'Variables por división',
    min_samples_leaf: 'Mínimo por hoja',
  },
  hist_gradient_boosting: {
    max_iter: 'Iteraciones', learning_rate: 'Tasa de aprendizaje', max_leaf_nodes: 'Máximo de hojas',
    max_depth: 'Profundidad máxima', min_samples_leaf: 'Mínimo por hoja', l2_regularization: 'Regularización L2',
  },
}

function resumenConfiguracion(config: Configuracion, tipo?: TipoMaxFeatures, otras: Configuracion[] = []): string {
  const principales = PARAMETROS_VISIBLES[config.tecnica]
  const variantes = otras.filter(otra => otra.tecnica === config.tecnica)
  const partes = Object.entries(config.parametros).filter(([clave, valor]) =>
    Object.hasOwn(principales, clave) || variantes.some(otra => otra.parametros[clave] !== valor)
  ).sort(([a], [b]) => a.localeCompare(b)).map(([clave, valor]) => {
    let visible = valor === null ? 'sin límite' : valor === true ? 'sí' : valor === false ? 'no' : String(valor)
    if (clave === 'max_features' && config.tecnica === 'random_forest' && typeof valor === 'number' && !tipo) {
      visible = 'tipo indeterminado'
    }
    if (clave === 'max_features' && tipo === 'real' && typeof valor === 'number') {
      visible = new Intl.NumberFormat('es-EC', { style: 'percent', maximumSignificantDigits: 6 }).format(valor)
    }
    return `${principales[clave] ?? clave.replaceAll('_', ' ')}: ${visible}`
  })
  if (config.variables.length > 0) {
    partes.unshift(config.version_contrato === 1 ? 'Variables clásicas' : 'Variables ampliadas')
    if (variantes.some(otra => otra.version_contrato === config.version_contrato &&
      !mismasFechas(otra.variables, config.variables))) partes.push(`Orden: ${config.variables.join(', ')}`)
  }
  return partes.join(' · ')
}

function periodo(corte: Corte): string {
  const mostrar = (paso: number) => new Intl.DateTimeFormat('es-EC', {
    day: 'numeric', month: 'short', timeZone: 'UTC',
  }).format(new Date(`${fechaObjetivo(corte, paso)}T00:00:00Z`))
  return `${mostrar(1)} al ${mostrar(15)} de 2017`
}

function PronosticosAyuda() {
  return <p>En <strong>Pronósticos</strong>, genere manualmente los pronósticos de validación para la misma tienda, familia y
    configuración. Use los cortes indicados y vuelva aquí a actualizar el listado.</p>
}

function RevisionDisponibilidad({ opciones, analisis, resumen, analizar, cancelar }: {
  opciones: Opcion[]
  analisis: Analisis
  resumen: ReturnType<typeof resumirAnalisis> | null
  analizar: () => void
  cancelar: () => void
}) {
  const botonBuscar = useRef<HTMLButtonElement>(null)
  const conteos = CORTES.map(corte => ({ corte, cantidad: opciones.filter(p => p.fecha_corte === corte).length }))
  const minimo = Math.min(...conteos.map(c => c.cantidad))
  const completos = resumen?.grupos.filter(g => g.cortes.size === CORTES.length).length ?? 0
  const incidencias = resumen?.incidencias ?? []
  const limite = incidencias.length === 0 && resumen ? completos : minimo
  const idsSinVerificar = new Set(incidencias.map(i => i.id))
  const sinVerificar = opciones.filter(p => idsSinVerificar.has(p.id))
  return <>
    <div className="evaluation-actions">
      <button className="consolidation-primary" ref={botonBuscar} type="button"
        disabled={opciones.length === 0} onClick={analizar}>
        {analisis.kind === 'loading' ? 'Reiniciar búsqueda' : 'Buscar configuraciones para comparar'}
      </button>
      {analisis.kind === 'loading' && <button type="button" onClick={() => {
        cancelar()
        botonBuscar.current?.focus()
      }}>Cancelar análisis</button>}
    </div>
    {analisis.kind === 'loading' && <p role="status">Revisando {opciones.length} pronósticos guardados…</p>}
    {analisis.kind === 'cancelado' && <p role="status">Búsqueda cancelada. Puede volver a iniciarla.</p>}
    {limite < 2 && <p className="evaluation-notice"><strong>Insuficiencia comprobada para comparar dos configuraciones.</strong>{' '}
      {minimo < 2 ? 'Algún período tiene menos de dos pronósticos de validación.' :
        'Cada configuración necesita un pronóstico para cada uno de los tres períodos.'}</p>}
    {conteos.some(c => c.cantidad === 0) && <p>Cortes sin pronósticos de validación: {conteos.filter(c => c.cantidad === 0)
      .map(c => c.corte).join(', ')}.</p>}
    {incidencias.length > 0 && <div className="evaluation-notice">
      <p role="alert"><strong>Búsqueda incompleta.</strong> No se pudo comprobar la configuración de {incidencias.length} de {opciones.length} pronósticos.
        Puede haber más configuraciones o períodos disponibles. Vuelva a buscar o actualice el listado.</p>
      <details className="consolidation-details">
        <summary>Ver incidencias del análisis</summary>
        <ul className="consolidation-candidates">{incidencias.map(i => <li key={i.id}>Pronóstico {i.id}: {i.mensaje}</li>)}</ul>
      </details>
    </div>}
    {resumen && <p role="status"><strong>{completos} {completos === 1 ? 'configuración identificada' :
      'configuraciones identificadas'} con los tres períodos.</strong> {completos >= 2 && 'Puede elegirlas en el paso 2.'}</p>}
    {resumen?.grupos.map(({ referente, cortes }) => {
      const config = referente.datos.configuracion
      const pendientes = new Set(sinVerificar.filter(p => p.tecnica === config.tecnica && !cortes.has(p.fecha_corte)).map(p => p.fecha_corte))
      const faltantes = CORTES.filter(corte => !cortes.has(corte) && !pendientes.has(corte))
      return <article className="consolidation-availability" key={referente.datos.pronostico_id}>
        <div className="consolidation-availability-heading">
          <h5>{TECNICAS[config.tecnica]}</h5>
          <strong>{cortes.size} de 3 períodos {pendientes.size === 0 ? 'disponibles' : 'confirmados'}</strong>
        </div>
        <p className="consolidation-parameters">{resumenConfiguracion(config, referente.tipoMaxFeatures,
          resumen.grupos.map(g => g.referente.datos.configuracion))}</p>
        <ul className="consolidation-periods">{CORTES.map(corte => <li key={corte}>
          <span>{periodo(corte)} <small>· corte {corte}</small></span>
          <strong>{cortes.has(corte) ? 'Disponible' : pendientes.has(corte) ? 'Por verificar' : 'Falta pronóstico'}</strong>
        </li>)}</ul>
        {faltantes.length > 0 && <p>Para {faltantes.map(periodo).join(' y ')}, genere manualmente un pronóstico de validación en
          {' '}<strong>Pronósticos</strong> con {faltantes.length === 1 ? 'el corte' : 'los cortes'} {faltantes.join(', ')}.
          Use la misma tienda y familia, y conserve la técnica, las variables y los parámetros de esta configuración.</p>}
        {pendientes.size > 0 && <p>Vuelva a buscar para comprobar los períodos marcados «Por verificar».</p>}
        <details className="consolidation-details">
          <summary>Ver parámetros completos y candidatos</summary>
          <ul className="consolidation-candidates">{CORTES.map(corte => <li key={corte}>
            <time dateTime={corte}>{corte}</time> · IDs disponibles: {cortes.get(corte)?.join(', ') ?? '—'}
          </li>)}</ul>
          {referente.tipoMaxFeatures && <p>Tipo de max_features: {referente.tipoMaxFeatures === 'entero' ? 'conteo entero' : 'fracción'}.</p>}
          <ConfiguracionPublica datos={config} />
        </details>
      </article>
    })}
    {limite < 2 && !resumen?.grupos.length && <PronosticosAyuda />}
    <details className="consolidation-details">
      <summary>Ver conteo de pronósticos por período</summary>
      <p>Total elegible: {opciones.length}. Para dos o tres grupos hacen falta dos o tres pronósticos distintos por período, respectivamente.</p>
      <ul className="consolidation-candidates">{conteos.map(({ corte, cantidad }) => <li key={corte}>
        {periodo(corte)} · corte {corte} · {cantidad} {cantidad === 1 ? 'candidato' : 'candidatos'} de validación
      </li>)}</ul>
    </details>
  </>
}

function candidatosPreparados(borrador: Borrador[], opciones: Opcion[]): Grupo[] {
  return borrador.map(({ configuracion, candidatos }) => CORTES.map(corte => {
    const ids = configuracion.cortes.get(corte) ?? []
    const id = ids.length === 1 ? ids[0] : candidatos[corte]
    return opciones.find(p => p.id === id && p.fecha_corte === corte && ids.includes(p.id)) ?? null
  }))
}

function recuperarSeleccion(preparadas: ConfiguracionAnalizada[], grupos: Grupo[], completas: Disponibilidad[]): Borrador[] {
  return preparadas.flatMap((referente, i) => {
    const configuracion = completas.find(g => mismaConfiguracionPublica(g.referente, referente))
    if (!configuracion) return []
    const candidatos: Borrador['candidatos'] = {}
    CORTES.forEach((corte, j) => {
      const id = grupos[i]?.[j]?.id
      if (id !== undefined && configuracion.cortes.get(corte)?.includes(id)) candidatos[corte] = id
    })
    return [{ configuracion, candidatos }]
  })
}

const formatoFechaCandidato = new Intl.DateTimeFormat('es-EC', { dateStyle: 'long', timeZone: 'UTC' })

function etiquetaCandidato(p: Opcion): string {
  const fechaGuardada = p.fecha_generacion.slice(0, 10)
  const fecha = formatoFechaCandidato.format(new Date(`${fechaGuardada}T00:00:00Z`))
  const hora = p.fecha_generacion.slice(10).replace(/^T/, ' ')
  return `${fecha}${hora} · pronóstico ${p.id} · experimento ${p.experimento_id}`
}

function SeleccionGuiada({ completas, opciones, borrador, preparada, elegir, elegirCandidato }: {
  completas: Disponibilidad[]
  opciones: Opcion[]
  borrador: Borrador[]
  preparada: boolean
  elegir: (configuracion: Disponibilidad) => void
  elegirCandidato: (configuracion: Disponibilidad, corte: Corte, id: number) => void
}) {
  const variantes = completas.map(g => g.referente.datos.configuracion)
  const ayudaCandidatos = useId()
  const hayAlternativas = borrador.some(b => CORTES.some(corte => (b.configuracion.cortes.get(corte)?.length ?? 0) > 1))
  return <>
    <p>{preparada ? 'Selección preparada. Estas configuraciones se usarán en el paso 3.' :
      'Elija dos o tres configuraciones con los tres períodos disponibles.'}</p>
    {hayAlternativas && <p id={ayudaCandidatos} className="consolidation-parameters">
      Los pronósticos guardados para este mismo período comparten la configuración predictiva: técnica, variables y parámetros.
      Pueden haberse generado en fechas distintas o con distintas versiones del software. Use la fecha para distinguirlos.
      La compatibilidad se comprobará al consultar la consolidación.
    </p>}
    <div className="consolidation-choices">{completas.map(configuracion => {
      const { referente, cortes } = configuracion
      const config = referente.datos.configuracion
      const elegido = borrador.find(b => b.configuracion === configuracion)
      return <div className="consolidation-choice" key={referente.datos.pronostico_id}>
        <label className="consolidation-choice-label">
          <input type="checkbox" checked={Boolean(elegido)} disabled={preparada || (!elegido && borrador.length === 3)}
            onChange={() => elegir(configuracion)} />
          <span><strong>{TECNICAS[config.tecnica]}</strong>
            <span className="consolidation-parameters">{resumenConfiguracion(config, referente.tipoMaxFeatures, variantes)}</span></span>
        </label>
        {elegido && CORTES.map(corte => {
          const ids = cortes.get(corte) ?? []
          return ids.length > 1 && <label className="consolidation-candidate" key={corte}>
            <span>{periodo(corte)} · {preparada ? 'fecha de generación del pronóstico preparado' : 'elija un pronóstico por su fecha de generación'}</span>
            <select aria-describedby={ayudaCandidatos} disabled={preparada} value={elegido.candidatos[corte] ?? ''} onChange={e => elegirCandidato(configuracion, corte, Number(e.target.value))}>
              <option value="">Seleccione un candidato</option>
              {opciones.filter(p => ids.includes(p.id)).map(p => <option key={p.id} value={p.id}>{etiquetaCandidato(p)}</option>)}
            </select>
          </label>
        })}
      </div>
    })}</div>
    <p role="status">{preparada ? `${borrador.length} configuraciones preparadas.` :
      `${borrador.length} de 3 configuraciones seleccionadas.`}</p>
  </>
}

function Resultados({ datos, tiposMaxFeatures }: Extract<Consulta, { kind: 'success' }>) {
  const variantes = datos.resultados.map(g => g.configuracion)
  return <section className="consolidation-results" aria-label="Resultado de la consolidación">
    <h5>Resumen de los tres períodos</h5>
    <p>Conjunto {datos.conjunto_id} · tienda {datos.serie.store_nbr} · {datos.serie.family}.</p>
    <div className="evaluation-notice" role="status"><strong>{datos.observaciones_evaluadas_total}/45 observaciones · cobertura {datos.estado_cobertura.replaceAll('_', ' ')}.</strong>
      <p>{ALCANCES[datos.estado_cobertura]}</p>
      {datos.fechas_ausentes_por_corte.some(c => c.fechas_ausentes.length > 0) && <ul className="consolidation-dates">
        {datos.fechas_ausentes_por_corte.map(c => <li key={c.fecha_corte}>Corte {c.fecha_corte}: {c.fechas_ausentes.length} fechas sin registro
          {c.fechas_ausentes.length > 0 ? ` — ${c.fechas_ausentes.join(', ')}` : ''}.</li>)}</ul>}
    </div>
    <div className="consolidation-summary-cards">
      {datos.resultados.map(({ identidad_configuracion, tecnica, configuracion, consolidado }, i) =>
        <article className="consolidation-summary-card" key={identidad_configuracion}>
          <h6>{TECNICAS[tecnica]}</h6>
          <p className="consolidation-parameters">{resumenConfiguracion(configuracion, tiposMaxFeatures[i], variantes)}</p>
          <dl>
            <div><dt>Cobertura</dt><dd>{consolidado.observaciones_evaluadas}/45 días</dd></div>
            <div><dt>MAE</dt><dd>{consolidado.mae === null ? 'No definido' : presentarNumero(consolidado.mae)}</dd></div>
            <div><dt>WAPE</dt><dd>{consolidado.wape === null ? 'No definido' : `${presentarNumero(consolidado.wape)} %`}</dd></div>
            <div><dt>RMSE</dt><dd>{consolidado.rmse === null ? 'No definido' : presentarNumero(consolidado.rmse)}</dd></div>
          </dl>
        </article>
      )}
    </div>
    <p>Compare los errores sobre la cobertura indicada. Un MAE menor significa que el pronóstico se alejó menos de las ventas
      reales en promedio. El WAPE expresa el error en relación con el total vendido. El RMSE da más peso a los errores grandes.</p>
    <details className="consolidation-details">
      <summary>Ver métricas completas y resultados por corte</summary>
      <div className="evaluation-scroll consolidation-scroll" role="region" tabIndex={0} aria-label="Resumen consolidado desplazable">
        <table><caption>Métricas consolidadas · mismo orden que los grupos seleccionados</caption>
          <thead><tr><th scope="col">Configuración</th><th scope="col">Cobertura</th><th scope="col">MAE</th>
            <th scope="col">WAPE (%)</th><th scope="col">RMSE</th><th scope="col">Observaciones</th><th scope="col">Suma de errores absolutos</th><th scope="col">Suma de ventas reales</th></tr></thead>
          <tbody>{datos.resultados.map((g, i) => <tr key={g.identidad_configuracion}><th scope="row">Grupo {i + 1} · {TECNICAS[g.tecnica]}<small className="consolidation-parameters">
            {resumenConfiguracion(g.configuracion, tiposMaxFeatures[i], variantes)}</small></th>
            <td><Valor valor={g.consolidado.cobertura} /> (proporción)</td>
            <td><Valor valor={g.consolidado.mae} /></td><td><Valor valor={g.consolidado.wape} /></td><td><Valor valor={g.consolidado.rmse} /></td>
            <td>{g.consolidado.observaciones_evaluadas}/45</td>
            <td><Valor valor={g.consolidado.suma_errores_absolutos} /></td><td><Valor valor={g.consolidado.suma_ventas_reales} /></td></tr>)}</tbody>
        </table>
      </div>
      <p>El WAPE consolidado usa los totales de los tres períodos. El WAPE aparece como «No definido» si las ventas reales suman cero o no hay registros.</p>
      <div className="evaluation-scroll consolidation-scroll" role="region" tabIndex={0} aria-label="Métricas por corte desplazables">
        <table><caption>Resultados individuales por configuración y corte</caption>
          <thead><tr><th scope="col">Grupo y técnica</th><th scope="col">Corte</th><th scope="col">Pronóstico / experimento</th>
            <th scope="col">MAE</th><th scope="col">WAPE (%)</th><th scope="col">RMSE</th><th scope="col">Observaciones</th>
            <th scope="col">Cobertura</th><th scope="col">Fechas sin registro</th></tr></thead>
          <tbody>{datos.resultados.flatMap((g, i) => g.evaluaciones.map(e => <tr key={e.pronostico_id}>
            <th scope="row">Grupo {i + 1} · {TECNICAS[g.tecnica]}</th><td>{e.fecha_corte}</td><td>{e.pronostico_id} / {e.experimento_id}</td>
            <td><Valor valor={e.evaluacion.mae} /></td><td><Valor valor={e.evaluacion.wape} /></td><td><Valor valor={e.evaluacion.rmse} /></td>
            <td>{e.evaluacion.observaciones_evaluadas}/15</td><td><Valor valor={e.evaluacion.cobertura} /> (proporción)</td>
            <td className="consolidation-dates">{e.evaluacion.fechas_sin_registro}{e.evaluacion.fechas_ausentes.length > 0 && ` — ${e.evaluacion.fechas_ausentes.join(', ')}`}</td>
          </tr>))}</tbody>
        </table>
      </div>
    </details>
    {datos.resultados.map((g, i) => <details className="consolidation-details" key={g.identidad_configuracion}>
      <summary>Grupo {i + 1} · {TECNICAS[g.tecnica]}: configuración y trazabilidad</summary>
      <ConfiguracionPublica datos={g.configuracion} />
      <p>Identidad predictiva declarada: <code className="consolidation-identity">{g.identidad_configuracion}</code></p>
      <dl>{Object.entries(g.versiones_ejecucion).map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
      <ul>{g.evaluaciones.map(e => <li key={e.pronostico_id}>{e.fecha_corte} · pronóstico {e.pronostico_id} · experimento {e.experimento_id}</li>)}</ul>
      <p>Esta identidad describe la configuración predictiva; no demuestra pesos entrenados idénticos ni máquinas iguales.
        Las versiones mostradas son las declaradas y comprobadas por el backend.</p>
    </details>)}
    <h4>Detalle diario por corte</h4>
    {CORTES.map((corte, j) => <details className="consolidation-details" key={corte}>
      <summary>Corte {corte} · quince fechas, ventas y errores diarios</summary>
      <div className="evaluation-scroll consolidation-scroll" role="region" tabIndex={0} aria-label={`Detalle diario desplazable del corte ${corte}`}>
        <table><caption>Quince fechas objetivo posteriores al corte {corte}</caption>
          <thead><tr><th scope="col" rowSpan={2}>Paso</th><th scope="col" rowSpan={2}>Fecha objetivo</th><th scope="col" rowSpan={2}>Venta real</th>
            {datos.resultados.map((g, i) => <th scope="colgroup" colSpan={2} key={g.identidad_configuracion}>Grupo {i + 1} · {TECNICAS[g.tecnica]} · pronóstico {g.evaluaciones[j].pronostico_id}</th>)}</tr>
            <tr>{datos.resultados.flatMap((g, i) => [<th scope="col" key={`${i}-venta`}>Estimada · grupo {i + 1}</th>,
              <th scope="col" key={`${i}-error`}>Error absoluto · grupo {i + 1} · pronóstico {g.evaluaciones[j].pronostico_id}</th>])}</tr></thead>
          <tbody>{datos.resultados[0].evaluaciones[j].evaluacion.detalles.map((d, k) => <tr key={d.paso}>
            <th scope="row">{d.paso}</th><td><time dateTime={d.fecha_objetivo}>{d.fecha_objetivo}</time></td><td><Valor valor={d.venta_real} ausente="Sin registro" /></td>
            {datos.resultados.flatMap((g, i) => [<td key={`${i}-venta`}><Valor valor={g.evaluaciones[j].evaluacion.detalles[k].venta_estimada} /></td>,
              <td key={`${i}-error`}><Valor valor={g.evaluaciones[j].evaluacion.detalles[k].error_absoluto} ausente="No evaluable" /></td>])}</tr>)}</tbody>
        </table>
      </div>
    </details>)}
  </section>
}

export default function ConsolidacionEvaluaciones({ seleccion }: Props) {
  const [lista, setLista] = useState<Lista>({ kind: 'loading' })
  const [grupos, setGrupos] = useState<Grupo[]>([[null, null, null], [null, null, null]])
  const [consulta, setConsulta] = useState<Consulta>({ kind: 'idle' })
  const [inspeccion, setInspeccion] = useState<Inspeccion>({ kind: 'idle' })
  const [analisis, setAnalisis] = useState<Analisis>({ kind: 'idle' })
  const [borrador, setBorrador] = useState<Borrador[] | null>(null)
  const [confirmarReemplazo, setConfirmarReemplazo] = useState(false)
  const [preparadas, setPreparadas] = useState<ConfiguracionAnalizada[] | null>(null)
  const resumen = analisis.kind === 'success' ? analisis.resumen : null
  const [revision, setRevision] = useState(0)
  const [aviso, setAviso] = useState('')
  const listado = useRef<AbortController | null>(null)
  const consolidacion = useRef<AbortController | null>(null)
  const vistaPrevia = useRef<AbortController | null>(null)
  const diagnostico = useRef<AbortController | null>(null)
  const pasoSeleccion = useRef<HTMLHeadingElement>(null)
  const pasoResultados = useRef<HTMLHeadingElement>(null)
  const identidad = useRef(0)
  const id = useId()
  const { conjuntoId, serie: { id: serieId, store_nbr: tienda, family: familia } } = seleccion

  useEffect(() => {
    const controller = new AbortController()
    listado.current = controller
    async function cargar() {
      setLista({ kind: 'loading' })
      try {
        const body = await obtener(`/api/pronosticos/active/series/${serieId}`, controller.signal)
        if (controller.signal.aborted || listado.current !== controller) return
        const datos = leerLista(body, { conjuntoId, serie: { id: serieId, store_nbr: tienda, family: familia } })
        if (!datos) throw new ErrorConsulta(RESPUESTA_INVALIDA)
        setGrupos(previos => previos.map(g => g.map(p => p !== null && datos.opciones.some(o => mismaOpcion(o, p)) ? p : null)))
        setLista({ kind: 'success', ...datos })
      } catch (error) {
        if (!controller.signal.aborted && listado.current === controller) setLista({ kind: 'error', mensaje: error instanceof ErrorConsulta ? error.message : RESPUESTA_INVALIDA })
      }
    }
    void cargar()
    return () => {
      controller.abort()
      consolidacion.current?.abort()
      vistaPrevia.current?.abort()
      diagnostico.current?.abort()
      diagnostico.current = null
    }
  }, [conjuntoId, serieId, tienda, familia, revision])

  function invalidar(mensaje = 'Selección actualizada; las consultas y resultados anteriores se descartaron.') {
    identidad.current++
    consolidacion.current?.abort()
    vistaPrevia.current?.abort()
    consolidacion.current = vistaPrevia.current = null
    setConsulta({ kind: 'idle' })
    setInspeccion({ kind: 'idle' })
    setAviso(mensaje)
  }
  function editarSeleccion() {
    setBorrador(recuperadas)
    setConfirmarReemplazo(false)
    setAviso('')
    pasoSeleccion.current?.focus()
  }

  function cancelarEdicion() {
    setBorrador(null)
    setConfirmarReemplazo(false)
  }

  function permitirEdicionManual() {
    return borrador === null && (!preparadas || window.confirm(
      '¿Cambiar a selección manual? Se quitará la preparación guiada y se conservarán los pronósticos que no cambie.',
    ))
  }

  function elegirConfiguracion(configuracion: Disponibilidad) {
    setConfirmarReemplazo(false)
    setBorrador(actual => {
      const seleccionada = actual ?? []
      return seleccionada.some(b => b.configuracion === configuracion) ? seleccionada.filter(b => b.configuracion !== configuracion) :
        seleccionada.length < 3 ? [...seleccionada, { configuracion, candidatos: {} }] : seleccionada
    })
  }

  function elegirCandidato(configuracion: Disponibilidad, corte: Corte, candidato: number) {
    setConfirmarReemplazo(false)
    setBorrador(actual => actual?.map(b => b.configuracion === configuracion ?
      { ...b, candidatos: { ...b.candidatos, [corte]: candidato } } : b) ?? null)
  }

  function preparar(confirmado: boolean) {
    if (lista.kind !== 'success' || analisis.kind !== 'success' || !borrador) return
    const siguientes = candidatosPreparados(borrador, lista.opciones)
    if (!gruposCompletos(siguientes, lista.opciones)) return
    if (grupos.some(g => g.some(p => p !== null)) && !confirmado) {
      setConfirmarReemplazo(true)
      return
    }
    invalidar('Pronósticos preparados. Ya puede consultar los resultados en el paso 3.')
    setGrupos(siguientes)
    setPreparadas(borrador.map(b => b.configuracion.referente))
    setConfirmarReemplazo(false)
    setBorrador(null)
    pasoResultados.current?.focus()
  }

  function elegir(grupo: number, posicion: number, valor: string) {
    if (lista.kind !== 'success') return
    const opcion = lista.opciones.find(p => String(p.id) === valor && p.fecha_corte === CORTES[posicion]) ?? null
    if (opcion && grupos.some((g, i) => g.some((p, j) => p?.id === opcion.id && (i !== grupo || j !== posicion)))) return
    if (!permitirEdicionManual()) return
    invalidar()
    setPreparadas(null)
    setConfirmarReemplazo(false)
    setGrupos(previos => previos.map((g, i) => i === grupo ? g.map((p, j) => j === posicion ? opcion : p) : g))
  }
  function cancelarAnalisis(estado: 'idle' | 'cancelado') {
    diagnostico.current?.abort()
    diagnostico.current = null
    setAnalisis({ kind: estado })
    setBorrador(actual => actual === null ? null : [])
    setConfirmarReemplazo(false)
  }
  async function iniciarAnalisis() {
    if (lista.kind !== 'success' || lista.opciones.length === 0) return
    // La acción se aplica una vez por estado del botón, incluso al reiniciar.
    if (diagnostico.current && (analisis.kind !== 'loading' || analisis.solicitud !== diagnostico.current)) return
    diagnostico.current?.abort()
    const controller = new AbortController()
    diagnostico.current = controller
    setAnalisis({ kind: 'loading', solicitud: controller })
    setBorrador(actual => actual === null ? null : [])
    setConfirmarReemplazo(false)
    try {
      const resultados = await analizarDisponibles(seleccion, lista.opciones, controller.signal)
      if (!controller.signal.aborted && diagnostico.current === controller) setAnalisis({ kind: 'success', resumen: resumirAnalisis(resultados) })
    } finally {
      if (diagnostico.current === controller) diagnostico.current = null
    }
  }
  async function inspeccionar(opcion: Opcion) {
    vistaPrevia.current?.abort()
    const controller = new AbortController()
    vistaPrevia.current = controller
    const solicitud = identidad.current
    setInspeccion({ kind: 'loading', opcion })
    try {
      const body = await obtener(`/api/pronosticos/active/${opcion.id}/configuracion`, controller.signal)
      if (controller.signal.aborted || vistaPrevia.current !== controller || identidad.current !== solicitud) return
      if (!esVistaPrevia(body, seleccion, opcion)) throw new ErrorConsulta(RESPUESTA_INVALIDA)
      setInspeccion({ kind: 'success', datos: body })
    } catch (error) {
      if (!controller.signal.aborted && vistaPrevia.current === controller && identidad.current === solicitud) {
        setInspeccion({ kind: 'error', mensaje: error instanceof ErrorConsulta ? error.message : RESPUESTA_INVALIDA })
      }
    } finally { if (vistaPrevia.current === controller) vistaPrevia.current = null }
  }
  async function consolidar() {
    if (!puedeConsultar || !gruposCompletos(grupos, opciones) || consolidacion.current) return
    const pedidos = grupos.map(g => g.map(p => ({ ...p })))
    const controller = new AbortController()
    consolidacion.current = controller
    const solicitud = identidad.current
    setConsulta({ kind: 'loading' })
    setAviso('')
    const parametros = new URLSearchParams()
    pedidos.forEach(g => parametros.append('grupo', g.map(p => p.id).join(',')))
    try {
      const tipos = new WeakMap<object, TipoMaxFeatures>()
      const body = await obtener(`/api/evaluaciones/consolidacion/active?${parametros}`, controller.signal,
        respuesta => leerRespuestaConTipos(respuesta, tipos))
      if (controller.signal.aborted || consolidacion.current !== controller || identidad.current !== solicitud) return
      if (!esConsolidacion(body, seleccion, pedidos)) throw new ErrorConsulta(RESPUESTA_INVALIDA)
      // El tipo ambiguo (como 1 frente a 1.0) debe acreditarse en esta misma respuesta.
      const tiposMaxFeatures = body.resultados.map(({ configuracion }) => {
        const valor = configuracion.parametros.max_features
        if (configuracion.tecnica !== 'random_forest' || typeof valor !== 'number') return undefined
        return tipos.get(configuracion.parametros) ?? (Number.isInteger(valor) ? undefined : 'real')
      })
      setConsulta({ kind: 'success', datos: body, tiposMaxFeatures })
    } catch (error) {
      if (!controller.signal.aborted && consolidacion.current === controller && identidad.current === solicitud) {
        setConsulta(error instanceof ErrorConsulta ?
          { kind: 'error', mensaje: error.message, detalleTecnico: error.detalleTecnico } :
          { kind: 'error', mensaje: RESPUESTA_INVALIDA })
      }
    } finally { if (consolidacion.current === controller) consolidacion.current = null }
  }
  const opciones = lista.kind === 'success' ? lista.opciones : []
  const completas = resumen?.grupos.filter(g => g.cortes.size === CORTES.length) ?? []
  const listaParaConsultar = lista.kind === 'success' && gruposCompletos(grupos, opciones)
  const borradorCompleto = gruposCompletos(candidatosPreparados(borrador ?? [], opciones), opciones)
  const recuperadas = preparadas ? recuperarSeleccion(preparadas, grupos, completas) : []
  const preparacionVigente = listaParaConsultar && recuperadas.length === grupos.length &&
    recuperadas.every((b, i) => CORTES.every((corte, j) => b.candidatos[corte] === grupos[i][j]?.id))
  const mostrarPreparada = borrador === null && preparacionVigente
  const puedeConsultar = borrador === null && listaParaConsultar && (!preparadas || preparacionVigente)
  const elegidos = new Set(grupos.flat().flatMap(p => p === null ? [] : [p.id]))
  return <section className="consolidation-panel" aria-label="Consolidación temporal">
    <p className="consolidation-intro">Compara el desempeño de diferentes configuraciones predictivas durante tres períodos de quince días.</p>
    <section className="consolidation-step" aria-labelledby={`${id}-paso1`}>
      <h4 id={`${id}-paso1`}>1. Revisar configuraciones disponibles</h4>
      {lista.kind === 'loading' && <p role="status">Cargando los pronósticos guardados…</p>}
      {lista.kind === 'error' && <p role="alert" className="evaluation-notice">{lista.mensaje}</p>}
      {lista.kind === 'success' && <>
        {lista.total === 0 ? <p>No hay pronósticos guardados para esta serie.</p> : opciones.length === 0 ?
          <p>La serie tiene registros guardados. Ninguno corresponde a los períodos de validación de esta comparación.</p> : null}
        <RevisionDisponibilidad opciones={opciones} analisis={analisis} resumen={resumen}
          analizar={() => { void iniciarAnalisis() }} cancelar={() => cancelarAnalisis('cancelado')} />
      </>}
      <div className="evaluation-actions"><button type="button" disabled={lista.kind === 'loading'} onClick={() => {
        invalidar('')
        cancelarAnalisis('idle')
        setLista({ kind: 'loading' })
        setRevision(r => r + 1)
      }}>Actualizar listado de pronósticos</button></div>
    </section>

    <section className="consolidation-step" aria-labelledby={`${id}-paso2`}>
      <h4 id={`${id}-paso2`} ref={pasoSeleccion} tabIndex={-1}>2. Elegir configuraciones para comparar</h4>
      {borrador !== null && <p className="evaluation-notice" role="status">
        <strong>Selección en edición.</strong> Prepare los cambios para poder consultarlos.
      </p>}
      {completas.length === 0 && <p>{resumen ? 'La búsqueda no identificó configuraciones con los tres períodos confirmados.' :
        'Busque las configuraciones disponibles en el paso 1.'}</p>}
      {completas.length === 1 && <p>Hay una configuración con los tres períodos. Para preparar la comparación necesita al menos dos.</p>}
      {completas.length >= 2 && <>
        <SeleccionGuiada completas={completas} opciones={opciones} borrador={mostrarPreparada ? recuperadas : borrador ?? []}
          preparada={mostrarPreparada} elegir={elegirConfiguracion} elegirCandidato={elegirCandidato} />
        {mostrarPreparada || (preparadas && borrador === null) ?
          <div className="evaluation-actions"><button type="button" onClick={editarSeleccion}>Editar selección</button></div> :
          <>
            {(borrador?.length ?? 0) >= 2 && !borradorCompleto && <p>Elija un pronóstico en cada período que ofrece varios candidatos.</p>}
            {confirmarReemplazo ? <div className="evaluation-notice" role="group" aria-label="Confirmar reemplazo">
              <p role="status">Ya hay pronósticos en los grupos. Confirme el reemplazo para usar la selección en edición.</p>
              <div className="evaluation-actions">
                <button className="consolidation-primary" type="button" onClick={() => preparar(true)}>Reemplazar y preparar pronósticos</button>
                <button type="button" onClick={cancelarEdicion}>Conservar grupos actuales</button>
              </div>
            </div> : <div className="evaluation-actions">
              <button className="consolidation-primary" type="button" disabled={!borradorCompleto} onClick={() => preparar(false)}>Preparar pronósticos</button>
            </div>}
          </>}
      </>}
      {borrador !== null && !confirmarReemplazo && <div className="evaluation-actions">
        <button type="button" onClick={cancelarEdicion}>Cancelar selección</button>
      </div>}
      {lista.kind === 'success' && <details className="consolidation-details consolidation-advanced">
        <summary>Selección manual avanzada</summary>
        <label className="consolidation-count" htmlFor={`${id}-grupos`}>Número de configuraciones
          <select id={`${id}-grupos`} value={grupos.length} disabled={borrador !== null} onChange={e => {
            const cantidad = Number(e.target.value)
            if ((cantidad !== 2 && cantidad !== 3) || !permitirEdicionManual()) return
            invalidar()
            setPreparadas(null)
            setConfirmarReemplazo(false)
            setGrupos(previos => cantidad === 2 ? previos.slice(0, 2) : [...previos.slice(0, 2), previos[2] ?? [null, null, null]])
          }}><option value={2}>Dos grupos</option><option value={3}>Tres grupos</option></select></label>
        <p id={`${id}-ayuda`}>Los grupos y cortes conservan este orden. Cada ID solo puede elegirse una vez.
          La inspección pública orienta la elección; no certifica compatibilidad entre los tres pronósticos.</p>
        <div className="consolidation-groups">{grupos.map((g, i) => <fieldset key={i} aria-describedby={`${id}-ayuda`} className="consolidation-group">
          <legend>Grupo {i + 1}</legend>{CORTES.map((c, j) => <div className="consolidation-slot" key={c}>
            <label htmlFor={`${id}-${i}-${j}`}>{j + 1}. Corte {c}</label>
            <select id={`${id}-${i}-${j}`} value={g[j]?.id ?? ''} disabled={borrador !== null} onChange={e => elegir(i, j, e.target.value)}>
              <option value="">Seleccione un pronóstico</option>{opciones.filter(p => p.fecha_corte === c).map(p =>
                <option key={p.id} value={p.id} disabled={elegidos.has(p.id) && g[j]?.id !== p.id}>{etiqueta(p)}</option>)}
            </select>
            {g[j] !== null && <p className="consolidation-selection">{etiqueta(g[j])}</p>}
            <button type="button" disabled={g[j] === null || (inspeccion.kind === 'loading' && inspeccion.opcion.id === g[j]?.id)}
              aria-label={`Inspeccionar configuración · grupo ${i + 1} · corte ${c}${g[j] ? ` · pronóstico ${g[j].id}` : ''}`}
              onClick={() => { const p = g[j]; if (p) void inspeccionar(p) }}>Inspeccionar configuración</button>
          </div>)}</fieldset>)}</div>
        <p role="status">{listaParaConsultar ? 'Selección lista: los grupos están completos. El backend verificará su compatibilidad.' :
          `Grupos incompletos: ${elegidos.size} de ${grupos.length * 3} selecciones. Se requieren IDs distintos para todas las posiciones.`}</p>
        {inspeccion.kind === 'loading' && <p role="status">Consultando configuración pública del pronóstico {inspeccion.opcion.id}…</p>}
        {inspeccion.kind === 'error' && <p role="alert" className="evaluation-notice">{inspeccion.mensaje}</p>}
        {inspeccion.kind === 'success' && <section className="consolidation-preview" aria-label="Configuración pública inspeccionada">
          <h4>Pronóstico {inspeccion.datos.pronostico_id} · experimento {inspeccion.datos.experimento_id} · corte {inspeccion.datos.fecha_corte}</h4>
          <p>{inspeccion.datos.es_predeterminada ? 'Configuración equivalente a la predeterminada.' : 'Configuración distinta de la predeterminada.'}
            {' '}Esta vista no acredita por sí sola la compatibilidad de un grupo.</p>
          <ConfiguracionPublica datos={inspeccion.datos.configuracion} />
        </section>}
      </details>}
    </section>

    <section className="consolidation-step" aria-labelledby={`${id}-paso3`}>
      <h4 id={`${id}-paso3`} ref={pasoResultados} tabIndex={-1}>3. Consultar resultados consolidados</h4>
      {borrador !== null ? <p>Hay una selección en edición. Prepárela o cancele la edición en el paso 2 para consultar.</p> :
        preparadas && !preparacionVigente ? <p role="status">{resumen ?
          'La búsqueda no pudo confirmar la selección anterior. Revise las configuraciones disponibles y prepare la selección.' :
          'Vuelva a buscar configuraciones en el paso 1 para verificar la selección preparada.'}</p> :
        listaParaConsultar ? <>
          <p><strong>{preparadas ? 'Selección preparada' : 'Selección manual completa'}.</strong> Se consultarán {grupos.length} configuraciones.</p>
          {mostrarPreparada && <ul className="consolidation-candidates">{recuperadas.map(({ configuracion }) => {
            const { datos, tipoMaxFeatures } = configuracion.referente
            return <li key={datos.pronostico_id}><strong>{TECNICAS[datos.configuracion.tecnica]}</strong>{' '}
              <span className="consolidation-parameters">{resumenConfiguracion(datos.configuracion, tipoMaxFeatures,
                recuperadas.map(b => b.configuracion.referente.datos.configuracion))}</span>
            </li>
          })}</ul>}
        </> : <p>Prepare los pronósticos en el paso 2 para consultar la comparación.</p>}
      {aviso && borrador === null && <p role="status" className="consolidation-feedback">{aviso}</p>}
      <div className="evaluation-actions"><button className="consolidation-primary" type="button"
        disabled={!puedeConsultar || consulta.kind === 'loading'} onClick={() => { void consolidar() }}>Consultar consolidación</button></div>
      {borrador === null && consulta.kind === 'loading' && <p role="status">Consultando los resultados…</p>}
      {puedeConsultar && consulta.kind === 'error' && <div className="evaluation-notice">
        <p role="alert">{consulta.mensaje}</p>
        {consulta.detalleTecnico && <details className="consolidation-details">
          <summary>Ver detalle técnico</summary>
          <p>{consulta.detalleTecnico}</p>
        </details>}
        {mostrarPreparada && <div className="evaluation-actions"><button type="button" onClick={editarSeleccion}>Editar selección rechazada</button></div>}
      </div>}
      {puedeConsultar && consulta.kind === 'success' && <Resultados {...consulta} />}
    </section>
  </section>
}
