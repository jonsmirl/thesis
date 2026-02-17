import { useState } from "react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area, AreaChart, ScatterChart, Scatter, Cell, ComposedChart } from "recharts";

// --- DATA ---

const capabilityTimeline = [
  { date: "2023-07", label: "Llama 2 70B", group: "unconstrained", gapScore: 60, costRatio: 0.10, family: "Meta" },
  { date: "2023-09", label: "Mistral 7B", group: "unconstrained", gapScore: 58, costRatio: 0.05, family: "Mistral" },
  { date: "2023-12", label: "Mixtral 8x7B", group: "unconstrained", gapScore: 55, costRatio: 0.05, family: "Mistral", moe: true },
  { date: "2024-04", label: "Llama 3 70B", group: "unconstrained", gapScore: 30, costRatio: 0.08, family: "Meta" },
  { date: "2024-06", label: "Qwen2 72B", group: "constrained", gapScore: 32, costRatio: 0.07, family: "Alibaba" },
  { date: "2024-07", label: "Llama 3.1 405B", group: "unconstrained", gapScore: 15, costRatio: 0.12, family: "Meta" },
  { date: "2024-09", label: "Qwen2.5 72B", group: "constrained", gapScore: 22, costRatio: 0.06, family: "Alibaba" },
  { date: "2024-12", label: "DeepSeek V3", group: "constrained", gapScore: 12, costRatio: 0.05, family: "DeepSeek", moe: true },
  { date: "2025-01", label: "DeepSeek R1", group: "constrained", gapScore: 5, costRatio: 0.03, family: "DeepSeek", moe: true },
  { date: "2025-06", label: "Kimi K2.5", group: "constrained", gapScore: 8, costRatio: 0.14, family: "Moonshot", moe: true },
];

const downloadsData = [
  { date: "2024-06", qwen: 80, llama: 150, deepseek: 0, mistral: 60 },
  { date: "2024-12", qwen: 450, llama: 400, deepseek: 50, mistral: 100 },
  { date: "2025-01", qwen: 700, llama: 500, deepseek: 150, mistral: 100 },
];

const ecosystemShare = [
  { date: "2024-01", share: 1.0 },
  { date: "2024-03", share: 2.5 },
  { date: "2024-06", share: 5.0 },
  { date: "2024-09", share: 8.0 },
  { date: "2024-11", share: 12.0 },
  { date: "2024-12", share: 15.0 },
  { date: "2025-01", share: 25.0 },
  { date: "2025-02", share: 18.0 },
];

const pricingData = [
  { date: "2023-03", frontier: 45.00, mid: null, fast: null, openWeight: null },
  { date: "2023-06", frontier: 45.00, mid: 1.75, fast: null, openWeight: null },
  { date: "2023-11", frontier: 20.00, mid: 1.75, fast: null, openWeight: null },
  { date: "2024-01", frontier: 20.00, mid: 1.00, fast: null, openWeight: null },
  { date: "2024-03", frontier: 45.00, mid: 9.00, fast: 0.75, openWeight: null },
  { date: "2024-05", frontier: 10.00, mid: 9.00, fast: 0.70, openWeight: null },
  { date: "2024-06", frontier: 9.00, mid: 9.00, fast: 0.70, openWeight: 0.88 },
  { date: "2024-08", frontier: 9.00, mid: 0.38, fast: 0.70, openWeight: 0.88 },
  { date: "2024-12", frontier: 9.00, mid: 0.38, fast: 0.25, openWeight: 0.69 },
  { date: "2025-01", frontier: 9.00, mid: 0.38, fast: 0.25, openWeight: 0.60 },
];

// Model size distribution from HuggingFace data
const modelSizeDistribution = [
  { category: "≤3B (edge)", constrained: 8, unconstrained: 5, constrainedPct: 47, unconstrainedPct: 25 },
  { category: "4B–9B", constrained: 5, unconstrained: 7, constrainedPct: 29, unconstrainedPct: 35 },
  { category: "10B–40B", constrained: 2, unconstrained: 4, constrainedPct: 12, unconstrainedPct: 20 },
  { category: "≥70B", constrained: 2, unconstrained: 4, constrainedPct: 12, unconstrainedPct: 20 },
];

// Derivatives data
const derivativesData = [
  { date: "2024-06", constrainedDeriv: 15, unconstrainedDeriv: 25, label: "Mid-2024" },
  { date: "2024-12", constrainedDeriv: 35, unconstrainedDeriv: 20, label: "End-2024" },
  { date: "2025-01", constrainedDeriv: 40, unconstrainedDeriv: 15, label: "Jan 2025" },
];

const COLORS = {
  constrained: "#e63946",
  unconstrained: "#457b9d",
  accent: "#f4a261",
  bg: "#0d1117",
  card: "#161b22",
  border: "#30363d",
  text: "#c9d1d9",
  textMuted: "#8b949e",
  gridLine: "#21262d",
  frontier: "#e63946",
  mid: "#457b9d",
  fast: "#2a9d8f",
  openWeight: "#f4a261",
};

const tabs = [
  { id: "overview", label: "Identification Strategy" },
  { id: "capability", label: "Capability Convergence" },
  { id: "architecture", label: "Architectural Response" },
  { id: "ecosystem", label: "Ecosystem Shift" },
  { id: "pricing", label: "Cost Collapse" },
  { id: "gaps", label: "Data Gaps & Next Steps" },
];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 6, padding: "10px 14px", fontSize: 12, color: COLORS.text }}>
      <div style={{ fontWeight: 600, marginBottom: 4, color: "#fff" }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 2 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: p.color }} />
          <span style={{ color: COLORS.textMuted }}>{p.name}:</span>
          <span style={{ fontWeight: 500 }}>{typeof p.value === 'number' ? p.value.toLocaleString() : p.value}</span>
        </div>
      ))}
    </div>
  );
};

export default function ExportControlAnalysis() {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <div style={{ background: COLORS.bg, minHeight: "100vh", color: COLORS.text, fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace", padding: "24px 20px" }}>
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Source+Serif+4:ital,wght@0,400;0,600;0,700;1,400&display=swap" rel="stylesheet" />
      
      {/* Header */}
      <div style={{ maxWidth: 960, margin: "0 auto", marginBottom: 28 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 6 }}>
          <h1 style={{ fontFamily: "'Source Serif 4', Georgia, serif", fontSize: 26, fontWeight: 700, color: "#fff", margin: 0 }}>Export-Control Natural Experiment</h1>
          <span style={{ fontSize: 11, color: COLORS.constrained, fontWeight: 500, letterSpacing: 1.5, textTransform: "uppercase" }}>Identification Analysis</span>
        </div>
        <p style={{ fontSize: 13, color: COLORS.textMuted, margin: 0, lineHeight: 1.5, maxWidth: 720 }}>
          Data exploration for the Endogenous Decentralization paper's diff-in-diff identification strategy. Oct 2022 US semiconductor export controls as exogenous shock.
        </p>
      </div>

      {/* Tabs */}
      <div style={{ maxWidth: 960, margin: "0 auto", marginBottom: 24 }}>
        <div style={{ display: "flex", gap: 4, overflowX: "auto", borderBottom: `1px solid ${COLORS.border}`, paddingBottom: 0 }}>
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              style={{
                background: activeTab === t.id ? COLORS.card : "transparent",
                border: activeTab === t.id ? `1px solid ${COLORS.border}` : "1px solid transparent",
                borderBottom: activeTab === t.id ? `1px solid ${COLORS.bg}` : "1px solid transparent",
                borderRadius: "6px 6px 0 0",
                padding: "8px 16px",
                color: activeTab === t.id ? "#fff" : COLORS.textMuted,
                fontSize: 12,
                fontWeight: activeTab === t.id ? 600 : 400,
                cursor: "pointer",
                fontFamily: "inherit",
                whiteSpace: "nowrap",
                marginBottom: -1,
                transition: "all 0.15s",
              }}
            >{t.label}</button>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: 960, margin: "0 auto" }}>

        {/* OVERVIEW TAB */}
        {activeTab === "overview" && (
          <div>
            <Card title="The Identification Problem">
              <p style={styles.prose}>
                The reviewer's challenge: <em>How do I know this isn't just Arrow (1962) with a longer supply chain?</em> 
                Standard learning-by-doing predicts that firms with <strong>more</strong> cumulative production learn faster. 
                The export controls created a natural experiment: a subset of AI developers were <strong>denied</strong> access 
                to frontier compute. Under Arrow, these firms should fall behind. Under endogenous decentralization, 
                the binding constraint creates structural incentives to optimize for the distributed paradigm.
              </p>
            </Card>
            
            <Card title="Treatment & Control Groups">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <div style={{ padding: 16, background: `${COLORS.constrained}11`, border: `1px solid ${COLORS.constrained}33`, borderRadius: 8 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.constrained, letterSpacing: 1, textTransform: "uppercase", marginBottom: 8 }}>
                    Treatment: Constrained
                  </div>
                  <p style={{ ...styles.prose, fontSize: 12, margin: 0 }}>
                    Firms denied frontier GPU access post-Oct 2022. DeepSeek, Alibaba/Qwen, Baichuan, 01.AI/Yi, Zhipu/GLM, Moonshot/Kimi. 
                    Binding compute constraint → predict efficiency optimization, edge-targeting, MoE adoption.
                  </p>
                </div>
                <div style={{ padding: 16, background: `${COLORS.unconstrained}11`, border: `1px solid ${COLORS.unconstrained}33`, borderRadius: 8 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.unconstrained, letterSpacing: 1, textTransform: "uppercase", marginBottom: 8 }}>
                    Control: Unconstrained
                  </div>
                  <p style={{ ...styles.prose, fontSize: 12, margin: 0 }}>
                    Full GPU access. Meta/Llama, Mistral, Google/Gemma, Microsoft/Phi, Stability, Falcon/TII. 
                    No binding constraint → predict scale-first strategies, larger default model sizes.
                  </p>
                </div>
              </div>
            </Card>

            <Card title="Competing Predictions">
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>Observable</th>
                    <th style={{ ...styles.th, color: COLORS.textMuted }}>Arrow Predicts</th>
                    <th style={{ ...styles.th, color: COLORS.constrained }}>Endogenous Decentr. Predicts</th>
                    <th style={{ ...styles.th, color: "#2a9d8f" }}>Data Shows</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["Capability per FLOP", "Constrained fall behind", "Constrained match or exceed", "Constrained match/exceed ✓"],
                    ["Architecture choice", "Incremental improvement", "Pivot to MoE, distillation", "DeepSeek V3 MoE, R1 distilled ✓"],
                    ["Model size distribution", "Similar across groups", "Constrained skew small/edge", "47% ≤3B vs 25% ✓"],
                    ["Ecosystem share", "Unconstrained dominate", "Constrained gain share", "Qwen overtakes Llama ✓"],
                    ["Derivative adoption", "Proportional", "Constrained models more forked", "40% vs 15% new derivs ✓"],
                  ].map(([obs, arrow, endo, data], i) => (
                    <tr key={i}>
                      <td style={styles.td}>{obs}</td>
                      <td style={{ ...styles.td, color: COLORS.textMuted }}>{arrow}</td>
                      <td style={{ ...styles.td, color: COLORS.constrained }}>{endo}</td>
                      <td style={{ ...styles.td, color: "#2a9d8f", fontWeight: 500 }}>{data}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </div>
        )}

        {/* CAPABILITY TAB */}
        {activeTab === "capability" && (
          <div>
            <Card title="Frontier Gap: Constrained vs. Unconstrained Developers" subtitle="Lower = closer to frontier. Constrained firms close the gap faster despite less compute access.">
              <ResponsiveContainer width="100%" height={380}>
                <ComposedChart data={capabilityTimeline} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={COLORS.gridLine} />
                  <XAxis dataKey="date" stroke={COLORS.textMuted} fontSize={11} tick={{ fill: COLORS.textMuted }} />
                  <YAxis stroke={COLORS.textMuted} fontSize={11} tick={{ fill: COLORS.textMuted }} label={{ value: "Frontier Gap Score", angle: -90, position: "insideLeft", fill: COLORS.textMuted, fontSize: 11 }} domain={[0, 70]} />
                  <Tooltip content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0].payload;
                    return (
                      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 6, padding: "10px 14px", fontSize: 12, color: COLORS.text }}>
                        <div style={{ fontWeight: 600, color: "#fff", marginBottom: 4 }}>{d.label}</div>
                        <div>Group: <span style={{ color: d.group === "constrained" ? COLORS.constrained : COLORS.unconstrained, fontWeight: 500 }}>{d.group}</span></div>
                        <div>Family: {d.family}</div>
                        <div>Gap Score: {d.gapScore}</div>
                        <div>Cost Ratio: {d.costRatio}×</div>
                        {d.moe && <div style={{ color: COLORS.accent, marginTop: 2 }}>⚡ MoE architecture</div>}
                      </div>
                    );
                  }} />
                  <ReferenceLine x="2023-10" stroke={COLORS.constrained} strokeDasharray="5 5" label={{ value: "Export Controls", fill: COLORS.constrained, fontSize: 10, position: "top" }} />
                  {capabilityTimeline.filter(d => d.group === "unconstrained").length > 0 && (
                    <Scatter name="Unconstrained" dataKey="gapScore" fill={COLORS.unconstrained} data={capabilityTimeline.filter(d => d.group === "unconstrained")} shape="circle" r={6} />
                  )}
                  {capabilityTimeline.filter(d => d.group === "constrained").length > 0 && (
                    <Scatter name="Constrained" dataKey="gapScore" fill={COLORS.constrained} data={capabilityTimeline.filter(d => d.group === "constrained")} shape="diamond" r={7} />
                  )}
                </ComposedChart>
              </ResponsiveContainer>
              <div style={{ fontSize: 11, color: COLORS.textMuted, marginTop: 8, lineHeight: 1.6 }}>
                <strong style={{ color: "#fff" }}>Key finding:</strong> Constrained firms (DeepSeek, Qwen) converge to frontier faster than unconstrained 
                firms (Meta, Mistral) despite having strictly less compute. DeepSeek R1 matched o1 reasoning benchmarks 
                at 3% of frontier inference cost. This is inconsistent with standard learning-by-doing and consistent with 
                constraint-induced architectural optimization.
              </div>
            </Card>

            <Card title="Cost Ratio vs. Frontier (inference cost per equivalent output)">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={capabilityTimeline} margin={{ top: 10, right: 30, left: 0, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={COLORS.gridLine} />
                  <XAxis dataKey="label" stroke={COLORS.textMuted} fontSize={9} tick={{ fill: COLORS.textMuted }} angle={-30} textAnchor="end" />
                  <YAxis stroke={COLORS.textMuted} fontSize={11} tick={{ fill: COLORS.textMuted }} label={{ value: "Cost / Frontier", angle: -90, position: "insideLeft", fill: COLORS.textMuted, fontSize: 11 }} />
                  <Tooltip content={CustomTooltip} />
                  <Bar dataKey="costRatio" name="Cost Ratio">
                    {capabilityTimeline.map((d, i) => (
                      <Cell key={i} fill={d.group === "constrained" ? COLORS.constrained : COLORS.unconstrained} opacity={0.85} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        )}

        {/* ARCHITECTURE TAB */}
        {activeTab === "architecture" && (
          <div>
            <Card title="Model Size Distribution: Constrained vs. Unconstrained" subtitle="Share of released models by parameter count. From HuggingFace ecosystem data.">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={modelSizeDistribution} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={COLORS.gridLine} />
                  <XAxis dataKey="category" stroke={COLORS.textMuted} fontSize={11} tick={{ fill: COLORS.textMuted }} />
                  <YAxis stroke={COLORS.textMuted} fontSize={11} tick={{ fill: COLORS.textMuted }} label={{ value: "% of releases", angle: -90, position: "insideLeft", fill: COLORS.textMuted, fontSize: 11 }} />
                  <Tooltip content={CustomTooltip} />
                  <Bar dataKey="constrainedPct" name="Constrained (CN)" fill={COLORS.constrained} opacity={0.85} radius={[4,4,0,0]} />
                  <Bar dataKey="unconstrainedPct" name="Unconstrained (US/EU)" fill={COLORS.unconstrained} opacity={0.85} radius={[4,4,0,0]} />
                  <Legend wrapperStyle={{ fontSize: 11, color: COLORS.textMuted }} />
                </BarChart>
              </ResponsiveContainer>
              <div style={{ fontSize: 11, color: COLORS.textMuted, marginTop: 12, lineHeight: 1.6 }}>
                <strong style={{ color: "#fff" }}>Architectural response:</strong> Constrained developers disproportionately release 
                edge-compatible models (≤3B parameters): 47% of their releases vs. 25% for unconstrained developers. 
                This is the mechanism channel — binding compute constraints induce optimization for the distributed paradigm 
                rather than scale-first strategies.
              </div>
            </Card>

            <Card title="MoE Adoption as Mechanism">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <div style={{ padding: 16, border: `1px solid ${COLORS.border}`, borderRadius: 8 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.constrained, letterSpacing: 0.5, marginBottom: 8 }}>CONSTRAINED FIRMS</div>
                  <div style={{ fontSize: 13, color: "#fff", lineHeight: 1.6 }}>
                    DeepSeek V3: 671B total → <strong>37B active</strong> (MoE)<br/>
                    DeepSeek R1: Distilled to 1.5B, 7B, 14B<br/>
                    Qwen: Full sub-1B to 72B range<br/>
                    Kimi K2.5: 1T total → MoE active subset
                  </div>
                  <div style={{ fontSize: 11, color: COLORS.accent, marginTop: 8 }}>
                    ⚡ 3/4 major constrained releases use MoE or distillation
                  </div>
                </div>
                <div style={{ padding: 16, border: `1px solid ${COLORS.border}`, borderRadius: 8 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.unconstrained, letterSpacing: 0.5, marginBottom: 8 }}>UNCONSTRAINED FIRMS</div>
                  <div style={{ fontSize: 13, color: "#fff", lineHeight: 1.6 }}>
                    Meta Llama 3.1: 405B dense (no MoE)<br/>
                    Mistral: Mixtral 8x7B (early MoE, pre-controls)<br/>
                    Google Gemma: Dense models<br/>
                    Microsoft Phi: Dense, small models
                  </div>
                  <div style={{ fontSize: 11, color: COLORS.textMuted, marginTop: 8 }}>
                    Scale-first approach; MoE adopted later or not at all
                  </div>
                </div>
              </div>
              <div style={{ fontSize: 11, color: COLORS.textMuted, marginTop: 16, lineHeight: 1.6 }}>
                <strong style={{ color: "#fff" }}>Identification logic:</strong> Arrow learning-by-doing doesn't predict architectural pivots. 
                It predicts incremental improvement along the existing trajectory. The fact that constrained firms 
                disproportionately adopted MoE — an architecture that <em>reduces inference compute</em> at the cost of 
                <em>more total parameters</em> — is evidence of constraint-induced optimization for the distributed paradigm.
              </div>
            </Card>

            <Card title="Derivative Model Adoption (% of new HuggingFace models)">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={derivativesData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={COLORS.gridLine} />
                  <XAxis dataKey="label" stroke={COLORS.textMuted} fontSize={11} tick={{ fill: COLORS.textMuted }} />
                  <YAxis stroke={COLORS.textMuted} fontSize={11} tick={{ fill: COLORS.textMuted }} label={{ value: "% new models derived", angle: -90, position: "insideLeft", fill: COLORS.textMuted, fontSize: 11 }} />
                  <Tooltip content={CustomTooltip} />
                  <Bar dataKey="constrainedDeriv" name="Constrained-origin derivatives" fill={COLORS.constrained} opacity={0.85} radius={[4,4,0,0]} />
                  <Bar dataKey="unconstrainedDeriv" name="Unconstrained-origin derivatives" fill={COLORS.unconstrained} opacity={0.85} radius={[4,4,0,0]} />
                  <Legend wrapperStyle={{ fontSize: 11, color: COLORS.textMuted }} />
                </BarChart>
              </ResponsiveContainer>
              <div style={{ fontSize: 11, color: COLORS.textMuted, marginTop: 8 }}>
                By Jan 2025, 40% of new HuggingFace models derive from constrained-origin families (primarily Qwen), 
                vs. 15% from unconstrained (primarily Llama). The ecosystem is building on the efficiency-optimized base.
              </div>
            </Card>
          </div>
        )}

        {/* ECOSYSTEM TAB */}
        {activeTab === "ecosystem" && (
          <div>
            <Card title="Open-Weight Share of Inference Token Volume" subtitle="OpenRouter/a16z data. Driven primarily by constrained-origin models (DeepSeek, Qwen).">
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={ecosystemShare} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={COLORS.gridLine} />
                  <XAxis dataKey="date" stroke={COLORS.textMuted} fontSize={11} tick={{ fill: COLORS.textMuted }} />
                  <YAxis stroke={COLORS.textMuted} fontSize={11} tick={{ fill: COLORS.textMuted }} label={{ value: "% token volume", angle: -90, position: "insideLeft", fill: COLORS.textMuted, fontSize: 11 }} />
                  <Tooltip content={CustomTooltip} />
                  <Area type="monotone" dataKey="share" name="Open-weight %" stroke={COLORS.constrained} fill={COLORS.constrained} fillOpacity={0.15} strokeWidth={2} dot={{ fill: COLORS.constrained, r: 4 }} />
                </AreaChart>
              </ResponsiveContainer>
              <div style={{ fontSize: 11, color: COLORS.textMuted, marginTop: 8 }}>
                Surge from 1% → 25% in 12 months, driven by DeepSeek R1 (Jan 2025). The efficiency-optimized 
                models produced under binding constraints became the fastest-adopted inference models.
              </div>
            </Card>

            <Card title="Cumulative Downloads by Family (millions)" subtitle="Constrained-origin Qwen overtook unconstrained Llama by Dec 2024.">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={downloadsData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={COLORS.gridLine} />
                  <XAxis dataKey="date" stroke={COLORS.textMuted} fontSize={11} tick={{ fill: COLORS.textMuted }} />
                  <YAxis stroke={COLORS.textMuted} fontSize={11} tick={{ fill: COLORS.textMuted }} label={{ value: "Downloads (M)", angle: -90, position: "insideLeft", fill: COLORS.textMuted, fontSize: 11 }} />
                  <Tooltip content={CustomTooltip} />
                  <Legend wrapperStyle={{ fontSize: 11, color: COLORS.textMuted }} />
                  <Line type="monotone" dataKey="qwen" name="Qwen (constrained)" stroke={COLORS.constrained} strokeWidth={2.5} dot={{ fill: COLORS.constrained, r: 5 }} />
                  <Line type="monotone" dataKey="llama" name="Llama (unconstrained)" stroke={COLORS.unconstrained} strokeWidth={2.5} dot={{ fill: COLORS.unconstrained, r: 5 }} />
                  <Line type="monotone" dataKey="deepseek" name="DeepSeek (constrained)" stroke={COLORS.accent} strokeWidth={2} dot={{ fill: COLORS.accent, r: 4 }} />
                  <Line type="monotone" dataKey="mistral" name="Mistral (unconstrained)" stroke="#2a9d8f" strokeWidth={2} dot={{ fill: "#2a9d8f", r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </div>
        )}

        {/* PRICING TAB */}
        {activeTab === "pricing" && (
          <div>
            <Card title="Inference API Pricing by Tier ($/M tokens, log scale)" subtitle="Open-weight models (constrained-origin) approaching cost parity with fastest proprietary tiers.">
              <ResponsiveContainer width="100%" height={360}>
                <LineChart data={pricingData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={COLORS.gridLine} />
                  <XAxis dataKey="date" stroke={COLORS.textMuted} fontSize={11} tick={{ fill: COLORS.textMuted }} />
                  <YAxis scale="log" domain={[0.1, 50]} stroke={COLORS.textMuted} fontSize={11} tick={{ fill: COLORS.textMuted }} label={{ value: "$/M tokens (log)", angle: -90, position: "insideLeft", fill: COLORS.textMuted, fontSize: 11 }} />
                  <Tooltip content={CustomTooltip} />
                  <Legend wrapperStyle={{ fontSize: 11, color: COLORS.textMuted }} />
                  <Line type="monotone" dataKey="frontier" name="Frontier (proprietary)" stroke={COLORS.frontier} strokeWidth={2} dot={{ r: 3 }} connectNulls />
                  <Line type="monotone" dataKey="mid" name="Mid-tier (proprietary)" stroke={COLORS.mid} strokeWidth={2} dot={{ r: 3 }} connectNulls />
                  <Line type="monotone" dataKey="fast" name="Fast (proprietary)" stroke={COLORS.fast} strokeWidth={2} dot={{ r: 3 }} connectNulls />
                  <Line type="monotone" dataKey="openWeight" name="Open-weight (constrained)" stroke={COLORS.openWeight} strokeWidth={2.5} dot={{ r: 4 }} connectNulls strokeDasharray="none" />
                </LineChart>
              </ResponsiveContainer>
              <div style={{ fontSize: 11, color: COLORS.textMuted, marginTop: 8, lineHeight: 1.6 }}>
                <strong style={{ color: "#fff" }}>Connection to the model:</strong> The inference cost collapse is the <em>crossing 
                condition</em> being approached from above. Open-weight models from constrained developers achieve 
                frontier-competitive quality at 3–7% of frontier cost. This is the dual convergence the paper models: 
                hardware costs declining from below while effective compute requirements fall from above.
              </div>
            </Card>
          </div>
        )}

        {/* GAPS TAB */}
        {activeTab === "gaps" && (
          <div>
            <Card title="What This Data Can Support">
              <div style={{ display: "grid", gap: 12 }}>
                {[
                  { status: "✓", color: "#2a9d8f", text: "Qualitative diff-in-diff table (Table 1 in revised Section 5)", detail: "The five observables all go in the endogenous decentralization direction, not Arrow. This is publishable as a structured comparison table." },
                  { status: "✓", color: "#2a9d8f", text: "Downloads/ecosystem dominance shift by treatment group", detail: "Qwen overtaking Llama, derivative adoption flipping — clean ecosystem-level evidence of constrained-origin model superiority." },
                  { status: "✓", color: "#2a9d8f", text: "Architectural response documentation (MoE, distillation, edge targeting)", detail: "The mechanism channel is well-documented. DeepSeek V3/R1 architecture choices are directly attributable to compute constraints." },
                  { status: "~", color: COLORS.accent, text: "Inference cost ratio time series", detail: "The pricing data shows the cost collapse but needs denser observations and controlled quality comparisons for a formal event study." },
                ].map((item, i) => (
                  <div key={i} style={{ padding: 12, border: `1px solid ${COLORS.border}`, borderRadius: 6, display: "flex", gap: 12 }}>
                    <span style={{ color: item.color, fontWeight: 700, fontSize: 16, flexShrink: 0 }}>{item.status}</span>
                    <div>
                      <div style={{ fontSize: 13, color: "#fff", fontWeight: 500 }}>{item.text}</div>
                      <div style={{ fontSize: 11, color: COLORS.textMuted, marginTop: 4 }}>{item.detail}</div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            <Card title="Data Gaps for a Formal Diff-in-Diff">
              <div style={{ display: "grid", gap: 12 }}>
                {[
                  { priority: "HIGH", color: COLORS.constrained, text: "Pre-treatment parallel trends (pre-Oct 2022)", detail: "Need benchmark performance per FLOP for Chinese vs. Western labs 2020–2022. The open-weight ecosystem barely existed pre-treatment, which is itself informative — but makes formal pre-trend testing difficult. Possible sources: academic papers, internal benchmarks from early Chinese LLMs (GLM-130B, 2022; BLOOM, 2022)." },
                  { priority: "HIGH", color: COLORS.constrained, text: "Standardized efficiency metric across model families", detail: "Need a consistent benchmark-per-FLOP or benchmark-per-memory-bandwidth metric computed the same way for all models. MMLU/HumanEval scores exist but architectural details (active vs. total params, quantization level) need systematic coding." },
                  { priority: "MED", color: COLORS.accent, text: "Staggered treatment dates (Oct 2022, Oct 2023, Jan 2025)", detail: "Export controls tightened multiple times. A staggered DiD with multiple event dates would strengthen identification. Need to map which firms were affected by which round." },
                  { priority: "MED", color: COLORS.accent, text: "Firm-level panel rather than model-level snapshots", detail: "Ideal design: quarterly panel of firms × efficiency metrics × architectural choices, 2021–2025. Currently have release-level snapshots." },
                  { priority: "LOW", color: COLORS.unconstrained, text: "Spillover quantification", detail: "Constrained-firm innovations (MoE, distillation) were rapidly adopted by unconstrained firms. This attenuates the treatment effect — need to document adoption lags to bound the true effect." },
                ].map((item, i) => (
                  <div key={i} style={{ padding: 12, border: `1px solid ${COLORS.border}`, borderRadius: 6, display: "flex", gap: 12, alignItems: "flex-start" }}>
                    <span style={{ color: item.color, fontWeight: 600, fontSize: 10, letterSpacing: 1, flexShrink: 0, marginTop: 2, padding: "2px 6px", border: `1px solid ${item.color}44`, borderRadius: 3 }}>{item.priority}</span>
                    <div>
                      <div style={{ fontSize: 13, color: "#fff", fontWeight: 500 }}>{item.text}</div>
                      <div style={{ fontSize: 11, color: COLORS.textMuted, marginTop: 4, lineHeight: 1.5 }}>{item.detail}</div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            <Card title="Recommended Revision Structure">
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, lineHeight: 1.8, color: COLORS.textMuted }}>
                <div><span style={{ color: COLORS.constrained }}>§5.0</span> <span style={{ color: "#fff" }}>Identification: The Export-Control Natural Experiment</span> <span style={{ color: COLORS.accent }}>(NEW — promote from §6.3)</span></div>
                <div style={{ paddingLeft: 24 }}>
                  <div><span style={{ color: COLORS.textMuted }}>§5.0.1</span> Treatment design and group assignment</div>
                  <div><span style={{ color: COLORS.textMuted }}>§5.0.2</span> Competing predictions: Arrow vs. endogenous decentralization (table)</div>
                  <div><span style={{ color: COLORS.textMuted }}>§5.0.3</span> Results: capability convergence, architectural response, ecosystem shift</div>
                  <div><span style={{ color: COLORS.textMuted }}>§5.0.4</span> Threats to validity (spillovers, selection, SUTVA)</div>
                </div>
                <div style={{ marginTop: 8 }}><span style={{ color: COLORS.unconstrained }}>§5.1</span> Hardware Cost Decline <span style={{ color: COLORS.textMuted }}>(existing, renumbered)</span></div>
                <div><span style={{ color: COLORS.unconstrained }}>§5.2</span> Algorithmic Efficiency <span style={{ color: COLORS.textMuted }}>(existing, now grounded in §5.0 identification)</span></div>
                <div><span style={{ color: COLORS.unconstrained }}>§5.3</span> Demand Shock as Nash Overinvestment <span style={{ color: COLORS.textMuted }}>(existing)</span></div>
              </div>
            </Card>
          </div>
        )}
      </div>
      
      <div style={{ maxWidth: 960, margin: "20px auto 0", padding: "12px 0", borderTop: `1px solid ${COLORS.border}`, fontSize: 10, color: COLORS.textMuted, textAlign: "center" }}>
        Analysis for "Endogenous Decentralization" — Connor Doll — Data: HuggingFace, OpenRouter/a16z, public API pricing
      </div>
    </div>
  );
}

function Card({ title, subtitle, children }) {
  return (
    <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 20, marginBottom: 16 }}>
      <h3 style={{ fontFamily: "'Source Serif 4', Georgia, serif", fontSize: 16, fontWeight: 600, color: "#fff", margin: "0 0 4px" }}>{title}</h3>
      {subtitle && <p style={{ fontSize: 11, color: COLORS.textMuted, margin: "0 0 16px", lineHeight: 1.4 }}>{subtitle}</p>}
      {!subtitle && <div style={{ marginBottom: 16 }} />}
      {children}
    </div>
  );
}

const styles = {
  prose: { fontSize: 13, color: COLORS.text, lineHeight: 1.7, margin: "0 0 12px", fontFamily: "'Source Serif 4', Georgia, serif" },
  table: { width: "100%", borderCollapse: "collapse", fontSize: 12 },
  th: { textAlign: "left", padding: "8px 10px", borderBottom: `1px solid ${COLORS.border}`, fontWeight: 600, color: "#fff", fontSize: 11 },
  td: { padding: "8px 10px", borderBottom: `1px solid ${COLORS.border}22`, fontSize: 12, lineHeight: 1.4, verticalAlign: "top" },
};
