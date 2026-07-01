import { DocumentPanel } from './components/DocumentPanel/DocumentPanel';
import { ChatPanel } from './components/ChatPanel/ChatPanel';
import './styles/global.css';

function App() {
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
          <DocumentPanel />
        </aside>
        <section className="app-chat">
          <ChatPanel />
        </section>
      </main>
    </div>
  );
}

export default App;
