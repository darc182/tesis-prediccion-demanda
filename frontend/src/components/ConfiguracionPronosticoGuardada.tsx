import { useId } from 'react'
import { identificarCandidato, type ResultadoConfiguracion, type useConfiguracionGuardada } from './consultaConfiguracionPronostico'

type Inspector = ReturnType<typeof useConfiguracionGuardada>

function ResumenConfiguracion({ pronostico, data }: ResultadoConfiguracion) {
  const { tecnica, parametros, variables, version_contrato, version_reglas, horizonte } = data.configuracion
  const base = tecnica === 'estacional_semanal_recursiva'
  const profundidad = parametros.max_depth === null ? 'sin límite' : String(parametros.max_depth)
  return <>
    <ul>
      <li>{base ? 'Línea base estacional' : version_contrato === 1 ? 'Variables clásicas' : 'Variables ampliadas'}</li>
      {base
        ? <li>Período estacional de {parametros.periodo_estacional_dias} días</li>
        : tecnica === 'random_forest' ? <>
          <li>Árboles: {parametros.n_estimators}</li>
          <li>Profundidad máxima: {profundidad}</li>
          <li>Mínimo de muestras por hoja: {parametros.min_samples_leaf}</li>
        </> : <>
          <li>Iteraciones: {parametros.max_iter}</li>
          <li>Tasa de aprendizaje: {parametros.learning_rate}</li>
          <li>Profundidad máxima: {profundidad}</li>
        </>}
      <li>{data.es_predeterminada ? 'Equivale a la configuración efectiva predeterminada.' :
        'Difiere de la configuración efectiva predeterminada.'}</li>
    </ul>
    <p className="help-text">Dos ejecuciones pueden compartir parámetros y haberse generado en fechas distintas.
      Las versiones internas de ejecución quedan fuera de esta consulta. La compatibilidad se verifica al consolidar.</p>
    <details>
      <summary>Consultar detalle técnico de la configuración</summary>
      <dl className="forecast-configuration-fields">
        <div><dt>Horizonte</dt><dd>{horizonte} días</dd></div>
        <div><dt>Versión del contrato</dt><dd>{version_contrato}</dd></div>
        <div><dt>Versión de reglas</dt><dd>{version_reglas}</dd></div>
        {pronostico.inicio_entrenamiento && <div><dt>Inicio de entrenamiento</dt><dd>{pronostico.inicio_entrenamiento}</dd></div>}
        {pronostico.fin_entrenamiento && <div><dt>Fin de entrenamiento</dt><dd>{pronostico.fin_entrenamiento}</dd></div>}
        {pronostico.duracion_segundos != null && <div><dt>Duración</dt><dd>{pronostico.duracion_segundos} s</dd></div>}
      </dl>
      <p>Variables predictoras {variables.length ? <code>{variables.join(', ')}</code> : '· No aplica'}</p>
      <h5>Parámetros efectivos</h5>
      <dl className="forecast-configuration-fields">
        {Object.entries(parametros).map(([nombre, valor]) => <div key={nombre}>
          <dt>{nombre === 'random_state' && 'Semilla aleatoria '}<code>{nombre}</code></dt><dd><code>{String(valor)}</code></dd>
        </div>)}
      </dl>
    </details>
  </>
}

export default function ConfiguracionPronosticoGuardada({ inspector, ubicacion = '' }: {
  inspector: Inspector; ubicacion?: string
}) {
  const id = useId()
  const { consulta, consultar, cerrar } = inspector
  if (!consulta || consulta.ubicacion !== ubicacion) return null
  const { estado, pronostico } = consulta
  return <section className="forecast-configuration" aria-labelledby={id}>
    <h4 id={id}>Configuración del pronóstico {pronostico.id}</h4>
    <p>{identificarCandidato(pronostico)}</p>
    {estado.kind === 'loading' && <p role="status">Consultando la configuración guardada…</p>}
    {estado.kind === 'error' && <>
      <p role="alert">{estado.message}</p>
      <button type="button" onClick={() => { void consultar(pronostico, ubicacion) }}>Reintentar consulta de configuración</button>{' '}
    </>}
    {estado.kind === 'success' && <ResumenConfiguracion pronostico={pronostico} data={estado.data} />}
    <button type="button" onClick={cerrar}>{estado.kind === 'loading' ? 'Cancelar consulta' : 'Cerrar configuración'}</button>
  </section>
}
