// FilterBar — dropdowns that scope a query by document metadata (M5).
//
// The available options are derived from the documents the user has actually
// uploaded, so we never show a filter value (e.g. a company) that would match
// nothing. Selecting "All" for a field clears that filter.

import { useMemo } from 'react';
import type { DocumentRecord, SearchFilters } from '../../types';

interface Props {
  documents: DocumentRecord[];
  filters: SearchFilters;
  onChange: (filters: SearchFilters) => void;
  disabled?: boolean;
}

// Collect the unique, sorted, non-empty values of one metadata field across
// all documents — these become a dropdown's options.
function uniqueValues<T>(values: (T | null | undefined)[]): T[] {
  const present = values.filter((v): v is T => v !== null && v !== undefined && v !== '');
  return Array.from(new Set(present)).sort();
}

export function FilterBar({ documents, filters, onChange, disabled }: Props) {
  const options = useMemo(
    () => ({
      companies: uniqueValues(documents.map(d => d.metadata.company)),
      years: uniqueValues(documents.map(d => d.metadata.year)),
      quarters: uniqueValues(documents.map(d => d.metadata.quarter)),
      docTypes: uniqueValues(documents.map(d => d.metadata.doc_type)),
    }),
    [documents],
  );

  // Update one field; an empty selection removes that key entirely.
  function update(field: keyof SearchFilters, value: string) {
    const next: SearchFilters = { ...filters };
    if (value === '') {
      delete next[field];
    } else if (field === 'year') {
      next.year = Number(value);
    } else {
      next[field] = value;
    }
    onChange(next);
  }

  const hasAnyFilter = Object.keys(filters).length > 0;

  // Nothing to filter on yet — hide the bar until documents exist.
  if (documents.length === 0) return null;

  return (
    <div className="filter-bar">
      <span className="filter-bar-label">Filter:</span>

      <select
        value={filters.company ?? ''}
        onChange={e => update('company', e.target.value)}
        disabled={disabled}
        title="Filter by company"
      >
        <option value="">All companies</option>
        {options.companies.map(c => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </select>

      <select
        value={filters.year ?? ''}
        onChange={e => update('year', e.target.value)}
        disabled={disabled}
        title="Filter by year"
      >
        <option value="">All years</option>
        {options.years.map(y => (
          <option key={y} value={y}>
            {y}
          </option>
        ))}
      </select>

      <select
        value={filters.quarter ?? ''}
        onChange={e => update('quarter', e.target.value)}
        disabled={disabled}
        title="Filter by quarter"
      >
        <option value="">All quarters</option>
        {options.quarters.map(q => (
          <option key={q} value={q}>
            {q}
          </option>
        ))}
      </select>

      <select
        value={filters.doc_type ?? ''}
        onChange={e => update('doc_type', e.target.value)}
        disabled={disabled}
        title="Filter by document type"
      >
        <option value="">All types</option>
        {options.docTypes.map(t => (
          <option key={t} value={t}>
            {t.replace(/_/g, ' ')}
          </option>
        ))}
      </select>

      {hasAnyFilter && (
        <button
          className="filter-bar-clear"
          onClick={() => onChange({})}
          disabled={disabled}
          title="Clear all filters"
        >
          Clear
        </button>
      )}
    </div>
  );
}
