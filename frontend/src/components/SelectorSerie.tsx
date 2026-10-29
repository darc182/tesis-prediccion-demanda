export type SerieTemporal = {
  id: number
  store_nbr: number
  family: string
}

export type SeriesState =
  | { kind: 'loading' }
  | { kind: 'network-error' | 'response-error'; message: string }
  | { kind: 'no-active' | 'empty' }
  | { kind: 'available'; conjuntoId: number; series: SerieTemporal[] }

type Props = {
  state: SeriesState
  selectedStore: number | null
  selectedFamily: string | null
  onStoreChange: (store: number | null) => void
  onFamilyChange: (family: string | null) => void
  onRetry: () => void
  idPrefix?: string
}

function SelectorSerie({ state, selectedStore, selectedFamily, onStoreChange, onFamilyChange, onRetry, idPrefix = '' }: Props) {
  const id = (nombre: string) => idPrefix ? `${idPrefix}-${nombre}` : nombre
  const series = state.kind === 'available' ? state.series : []
  const stores = [...new Set(series.map((serie) => serie.store_nbr))].sort((a, b) => a - b)
  const families = selectedStore === null ? [] : [
    ...new Set(series.filter((serie) => serie.store_nbr === selectedStore).map((serie) => serie.family)),
  ].sort((a, b) => a.localeCompare(b))

  return (
    <section className="series-card" aria-labelledby={id('series-heading')}>
      <h2 id={id('series-heading')}>Seleccionar serie temporal</h2>
      <p className="series-intro">Elija una tienda y una familia del conjunto de datos activo.</p>

      {state.kind === 'loading' && <p role="status">Cargando series disponibles…</p>}
      {(state.kind === 'network-error' || state.kind === 'response-error') && (
        <div role="alert">
          <p>{state.message}</p>
          <button type="button" className="retry-button" onClick={onRetry}>Reintentar consulta</button>
        </div>
      )}
      {state.kind === 'no-active' && <p>No hay un conjunto de datos activo. Cargue un archivo CSV para seleccionar una serie.</p>}
      {state.kind === 'empty' && <p>El conjunto activo no contiene series temporales disponibles.</p>}

      {state.kind === 'available' && (
        <>
          <p className="series-count">Conjunto activo {state.conjuntoId} · {series.length} series disponibles</p>
          <div className="series-filters">
            <div>
              <label htmlFor={id('series-store')}>Tienda</label>
              <select
                id={id('series-store')}
                value={selectedStore ?? ''}
                onChange={(event) => onStoreChange(event.target.value === '' ? null : Number(event.target.value))}
              >
                <option value="">Seleccione una tienda</option>
                {stores.map((store) => <option key={store} value={store}>{store}</option>)}
              </select>
            </div>
            <div>
              <label htmlFor={id('series-family')}>Familia de producto</label>
              <select
                id={id('series-family')}
                value={selectedFamily ?? ''}
                disabled={selectedStore === null}
                onChange={(event) => onFamilyChange(event.target.value || null)}
              >
                <option value="">Seleccione una familia</option>
                {families.map((family) => <option key={family} value={family}>{family}</option>)}
              </select>
            </div>
          </div>
          <p className="series-selection" role="status" aria-live="polite">
            {selectedStore !== null && selectedFamily !== null
              ? <>Serie seleccionada: <strong>tienda {selectedStore} · {selectedFamily}</strong></>
              : selectedStore !== null
                ? <>Tienda {selectedStore} seleccionada. Seleccione una familia.</>
                : 'Seleccione una tienda para ver sus familias disponibles.'}
          </p>
        </>
      )}
    </section>
  )
}

export default SelectorSerie
