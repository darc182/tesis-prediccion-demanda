import { useCallback, useLayoutEffect, useRef, useState } from 'react'
import type { SerieTemporal } from './SelectorSerie'

const NOMBRES = {
  estacional_semanal_recursiva: 'Línea base estacional semanal',
  random_forest: 'Random Forest',
  hist_gradient_boosting: 'HistGradientBoosting',
} as const
type Tecnica = keyof typeof NOMBRES
type Seleccion = { conjuntoId: number; serie: SerieTemporal }
export type CandidatoConfiguracion = {
  id: number; experimento_id: number; tecnica: Tecnica; fecha_corte: string
  horizonte: number; fecha_generacion: string
  inicio_entrenamiento?: string; fin_entrenamiento?: string; duracion_segundos?: number | null
}
export type ResultadoConfiguracion = { pronostico: CandidatoConfiguracion; data: ConfiguracionGuardada }
type EstadoConfiguracion = { kind: 'loading' } | { kind: 'error'; message: string } |
  { kind: 'success'; data: ConfiguracionGuardada }
type Consulta = {
  alcance: string; pronostico: CandidatoConfiguracion; ubicacion: string; estado: EstadoConfiguracion
}

export const VARIABLES_CLASICAS = ['lag_1', 'lag_7', 'lag_14', 'media_7', 'dia_semana', 'mes'] as const
export const VARIABLES_AMPLIADAS = [...VARIABLES_CLASICAS, 'media_14'] as const
type ParametroGuardado = number | string | boolean | null
type ConfiguracionGuardada = {
  es_predeterminada: boolean
  configuracion: {
    tecnica: Tecnica; parametros: Record<string, ParametroGuardado>; variables: string[]
    horizonte: 15; version_contrato: 1 | 2; version_reglas: 1 | 2
  }
}

const registro = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v)
function mismaSeleccion(value: Record<string, unknown>, seleccion: Seleccion): boolean {
  return registro(value.serie) && value.conjunto_id === seleccion.conjuntoId &&
    value.serie.id === seleccion.serie.id && value.serie.store_nbr === seleccion.serie.store_nbr &&
    value.serie.family === seleccion.serie.family
}
const conteoValido = (v: unknown): v is number => typeof v === 'number' && Number.isSafeInteger(v) && v >= 0

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
export const ERRORES_CONFIGURACION: Readonly<Record<string, string>> = {
  pronostico_inexistente: 'El pronóstico no existe en el conjunto activo.',
  pronostico_corrupto: 'El pronóstico almacenado está incompleto o presenta datos inconsistentes.',
  sin_conjunto_activo: 'No existe un conjunto activo para esta operación.',
  base_no_disponible: 'La base de datos no está disponible o no tiene el esquema necesario.',
  error_interno: 'Ocurrió un error interno al consultar la configuración.',
}

function esConfiguracionGuardada(value: unknown, seleccion: Seleccion, esperado: CandidatoConfiguracion): value is ConfiguracionGuardada {
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
export function instanteFecha(value: unknown): number | null {
  if (typeof value !== 'string' || !/^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(value)) return null
  const instante = Date.parse(`${value}T00:00:00Z`)
  return Number.isFinite(instante) && new Date(instante).toISOString().slice(0, 10) === value ? instante : null
}
export function fechaListadoValida(value: unknown): value is string {
  return typeof value === 'string' && value >= '0001-01-01' && instanteFecha(value) !== null
}
export function generacionListadoValida(value: unknown): value is string {
  if (typeof value !== 'string' || !fechaListadoValida(value.slice(0, 10))) return false
  const hora = /^(?:[T ]([0-9]{2}):([0-9]{2})(?::([0-9]{2})(?:\.[0-9]+)?)?(?:Z|[+-]([0-9]{2}):([0-9]{2}))?)?$/.exec(value.slice(10))
  return hora !== null && hora.slice(1).every((parte, i) =>
    parte === undefined || Number(parte) < (i === 0 || i === 3 ? 24 : 60)) && Number.isFinite(Date.parse(value))
}

const formatoFecha = new Intl.DateTimeFormat('es-EC', { dateStyle: 'medium', timeZone: 'UTC' })
export function mostrarFechaGeneracion(iso: string): string {
  const dia = formatoFecha.format(new Date(iso.slice(0, 10) + 'T00:00:00Z'))
  const hora = iso.slice(11)
  return hora ? dia + ' · ' + hora : dia
}
export function identificarCandidato(p: CandidatoConfiguracion): string {
  return NOMBRES[p.tecnica] + ' · Corte ' + p.fecha_corte + ' · Generado ' + mostrarFechaGeneracion(p.fecha_generacion) +
    ' · Pronóstico ' + p.id + ' · Experimento ' + p.experimento_id
}

class ErrorConfiguracion extends Error {}
async function obtenerConfiguracion(seleccion: Seleccion, candidato: CandidatoConfiguracion, signal: AbortSignal) {
  const respuesta = await fetch('/api/pronosticos/active/' + candidato.id + '/configuracion', { signal, cache: 'no-store' })
  let datos: unknown
  try { datos = await respuesta.json() } catch {
    throw new ErrorConfiguracion('El servidor respondió con HTTP ' + respuesta.status + ', pero no devolvió un JSON válido.')
  }
  if (!respuesta.ok) {
    const codigo = registro(datos) && registro(datos.detail) ? datos.detail.codigo : undefined
    const mensaje = typeof codigo === 'string' && Object.hasOwn(ERRORES_CONFIGURACION, codigo)
      ? ERRORES_CONFIGURACION[codigo] : 'La consulta de configuración falló.'
    throw new ErrorConfiguracion(mensaje + ' (HTTP ' + respuesta.status + ').')
  }
  if (respuesta.status !== 200 || !esConfiguracionGuardada(datos, seleccion, candidato)) {
    throw new ErrorConfiguracion('La configuración recibida no corresponde al pronóstico seleccionado o contiene datos incompatibles.')
  }
  return datos
}

// Un cambio de alcance invalida la respuesta y libera la solicitud, aunque el servidor
// termine después de la cancelación. Los filtros visuales no forman parte del alcance.
export function useConfiguracionGuardada(seleccion: Seleccion, alcance: string | null) {
  const [consulta, setConsulta] = useState<Consulta | null>(null)
  if (consulta !== null && consulta.alcance !== alcance) setConsulta(null)
  const solicitud = useRef<AbortController | null>(null)
  const abortar = useCallback(() => {
    solicitud.current?.abort()
    solicitud.current = null
  }, [])
  useLayoutEffect(() => abortar, [alcance, abortar])

  function cerrar() {
    abortar()
    setConsulta(null)
  }

  async function consultar(pronostico: CandidatoConfiguracion, ubicacion = '') {
    if (alcance === null) return
    abortar()
    const controlador = new AbortController()
    solicitud.current = controlador
    const consultaActual = { alcance, pronostico, ubicacion }
    setConsulta({ ...consultaActual, estado: { kind: 'loading' } })
    try {
      const data = await obtenerConfiguracion(seleccion, pronostico, controlador.signal)
      if (solicitud.current === controlador) setConsulta({ ...consultaActual, estado: { kind: 'success', data } })
    } catch (error) {
      if (solicitud.current === controlador) setConsulta({ ...consultaActual, estado: { kind: 'error', message:
        error instanceof ErrorConfiguracion ? error.message : 'No se pudo conectar para consultar la configuración guardada.' } })
    } finally {
      if (solicitud.current === controlador) solicitud.current = null
    }
  }

  return { consulta: consulta?.alcance === alcance ? consulta : null, consultar, cerrar }
}
