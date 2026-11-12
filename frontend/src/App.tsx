import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { useSpecs } from "./lib/api";
import { useApp } from "./lib/store";
import SpecsPage from "./pages/Specs";
import ArchitecturePage from "./pages/Architecture";
import TracePage from "./pages/Trace";
import ExperimentsPage from "./pages/Experiments";
import TrainingPage from "./pages/Training";
import LearnPage from "./pages/Learn";

export default function App() {
  const { model, setModel, theme, toggleTheme } = useApp();
  const specs = useSpecs();
  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">Inside Qwen-VL</span>
        <nav className="nav">
          {[["/specs", "Specs"], ["/architecture", "Architecture"], ["/trace", "Trace"], ["/experiments", "Experiments"], ["/training", "Training lab"], ["/learn", "Learn"]].map(([to, l]) => (
            <NavLink key={to} to={to} className={({ isActive }) => (isActive ? "active" : "")}>{l}</NavLink>
          ))}
        </nav>
        <div className="selectors">
          <label>Model
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {Object.entries(specs.data?.families ?? {}).map(([fam, f]) => (
                <optgroup key={fam} label={f.label}>
                  {specs.data!.models.filter((m) => m.family === fam).map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
                </optgroup>
              ))}
            </select>
          </label>
          <button onClick={toggleTheme} title="Toggle light/dark">{theme === "light" ? "☾ Dark" : "☀ Light"}</button>
        </div>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Navigate to="/architecture" replace />} />
          <Route path="/specs" element={<SpecsPage />} />
          <Route path="/architecture" element={<ArchitecturePage />} />
          <Route path="/trace" element={<TracePage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/learn" element={<LearnPage />} />
        </Routes>
      </main>
    </div>
  );
}
