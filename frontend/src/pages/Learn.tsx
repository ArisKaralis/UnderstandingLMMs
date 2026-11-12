import { GLOSSARY, REFERENCES } from "../content/glossary";

export default function LearnPage() {
  return (
    <div className="grid2">
      <div className="card">
        <h1>Glossary</h1>
        <table><tbody>{GLOSSARY.map((g) => (<tr key={g.term}><td style={{ verticalAlign: "top" }}><b>{g.term}</b></td><td style={{ whiteSpace: "normal" }}>{g.text}</td></tr>))}</tbody></table>
      </div>
      <div className="card">
        <h1>References</h1>
        <table><tbody>{REFERENCES.map((r) => (<tr key={r.url}><td style={{ whiteSpace: "normal" }}><a href={r.url} target="_blank" rel="noreferrer">{r.title}</a><div className="small muted">{r.note}</div></td></tr>))}</tbody></table>
        <h2 style={{ marginTop: 16 }}>Reading the dashboard</h2>
        <p>Architecture shows the model as a graph built from its real config; every node's shapes and operations are in drop-downs. Trace follows one real input through a captured forward pass. Experiments compares a baseline with an intervention. Training lab explains pre/post-training and replays tiny real SFT/RL runs step by step.</p>
      </div>
    </div>
  );
}
