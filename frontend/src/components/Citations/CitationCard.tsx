import type { Citation } from '../../types';

interface Props {
  citation: Citation;
  index: number;
}

export function CitationCard({ citation, index }: Props) {
  const matchPct = Math.round(citation.similarity_score * 100);
  const period = [citation.year, citation.quarter].filter(Boolean).join(' ');
  const metaBits = [citation.company, citation.ticker, period].filter(Boolean);

  return (
    <div className="citation-card">
      <div className="citation-header">
        <span className="citation-index">[{index}]</span>
        <span className="citation-source" title={citation.document_name}>
          {citation.document_name}
        </span>
        {citation.page_number != null && (
          <span className="citation-page">p.{citation.page_number}</span>
        )}
        <span className="citation-score">{matchPct}% match</span>
      </div>
      {metaBits.length > 0 && (
        <div className="citation-meta">
          {metaBits.map(bit => (
            <span key={bit} className="citation-chip">
              {bit}
            </span>
          ))}
        </div>
      )}
      <p className="citation-text">{citation.chunk_text}</p>
    </div>
  );
}
