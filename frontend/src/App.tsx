import { DocumentPanel } from './components/DocumentPanel/DocumentPanel';
import { ChatPanel } from './components/ChatPanel/ChatPanel';
import { useDocuments } from './hooks/useDocuments';
import './styles/global.css';

function App() {
  // Lifted here so both panels share one document list: the DocumentPanel
  // manages uploads, and the ChatPanel derives its filter options from the
  // same set — so a newly uploaded document immediately appears as a filter.
  const docs = useDocuments();

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-brand">
          <span className="app-header-icon">📊</span>
          <span className="app-header-title">Financial Research Assistant</span>
        </div>
        <span className="app-header-sub">RAG · Ollama · ChromaDB</span>
      </header>

      <main className="app-body">
        <aside className="app-sidebar">
          <DocumentPanel docs={docs} />
        </aside>
        <section className="app-chat">
          <ChatPanel documents={docs.documents} />
        </section>
      </main>
    </div>
  );
}

export default App;
