import { useState, useRef, useCallback } from "react";

const PAPER_CONTEXT = `PAPER SUMMARY: "The Distributed Stargate" argues $1.5-2T in AI datacenter investment is misallocated. Three converging trends will shift AI inference from centralized datacenters to billions of distributed consumer endpoints by ~2030:

(1) MODEL EFFICIENCY: Frontier quality achievable at 70-200B params by 2030 (vs 1-2T today) through MoE, distillation, test-time compute.

(2) MEMORY COSTS: HBM dies repackaged via SPHBM4 on organic substrates follow DRAM learning curves to $3-7/GB by 2030, enabling $500-1500 consumer inference devices.

(3) OPEN WEIGHTS: Chinese labs (DeepSeek, Qwen) release models at high benchmark scores for free.

CRITICAL NUANCE ON OPEN WEIGHTS AND MARKET DYNAMICS: Chinese open-weight models score well on benchmarks but are NOT YET dominating real-world usage outside China for serious professional tasks. GitHub AI-assisted commit data across all tools shows Claude Code leading by a wide margin for production coding — this is revealed preference, far more meaningful than benchmarks. Cursor, Windsurf, and similar tools overwhelmingly use Claude or GPT as backends. Open models currently fill the low-end: hobbyists, developing world, cost-sensitive applications, and local/private inference where data sovereignty matters.

HOWEVER — this bifurcation may NOT be stable. Two forces could tip the market toward open/distributed:

(A) PRICING PRESSURE: If Anthropic raises Claude's price (as they have been doing), the value proposition of "good enough" local inference on cheap hardware becomes compelling even for professional use cases. The gap between "95% as good for free" and "100% for increasing subscription/API fees" has an economic tipping point.

(B) HARDWARE COST COLLAPSE: The Rockchip RK1828 is a $25 bare chip with 5GB on-chip 3D-stacked DRAM, 20 TOPS INT8 compute, running 7B parameter models at 59-180 tokens/sec — FASTER THAN READING SPEED on a twenty-five dollar chip. The RK1820 (~$15-20) has 2.5GB for 3B models. These ship in SO-DIMM and M.2 form factors (plug into existing hardware). The next-gen RK1860 arrives Q2/Q3 2026. They already run Qwen2.5, Qwen3, and vision language models natively. This is the semiconductor cost curve thesis made concrete TODAY, not in 2030.

So the market is currently bifurcated — frontier closed models for high-value professional work, open models for the volume/low end — but the equilibrium is UNSTABLE. The question is: at what combination of (open model quality improvement) × (hardware cost decline) × (closed model price increase) does the professional segment tip? This is the core economic question.

EMPIRICAL ANCHORS: $25 RK1828 chip, 5GB on-chip DRAM, 180 tok/s on 7B models. Dev kits at $889-1029. Memory cost per GB on-chip: ~$5/GB (already below the paper's 2030 projection). GitHub AI commit data showing Claude Code market share. API pricing trends. HuggingFace download statistics by model family and geography.

Result: AI inference distributes from datacenters to billions of endpoints, paralleling mainframe-to-PC transition. Jevons Paradox means total compute demand increases but is funded by distributed consumer purchases. 30-50% of current infrastructure investment may be stranded — but the timeline and market structure are more complex than simple disruption narratives suggest.`;

const participants = {
  connor: {
    name: "Connor",
    color: "#1A5276",
    system: `You are Connor, a senior undergraduate studying Quantitative Economics at Tufts University. You are sharp, quantitatively rigorous, and not intimidated by famous economists. Your father wrote "The Distributed Stargate" paper (see context). You push for formal models, testable hypotheses, and quantitative rigor. You know editors need a formal model with testable predictions.

IMPORTANT: You understand the AI market well. Chinese open-weight models score well on benchmarks but Claude Code currently dominates professional coding workflows — GitHub AI-assisted commit data across all tools shows Claude Code leading by a wide margin in production code. This is revealed preference data.

BUT you recognize this bifurcation may be UNSTABLE. The Rockchip RK1828 — a $25 bare chip with 5GB on-chip 3D-stacked DRAM — runs 7B models at 59-180 tokens/sec, faster than reading speed. That's the 2030 cost curve arriving in 2025. If Anthropic keeps raising prices while $25 chips run increasingly capable open models at conversational speed, the professional segment could tip. The question isn't WHETHER the market is bifurcated today — it clearly is — but whether the equilibrium is stable. You want to model the tipping conditions formally: at what combination of (open model quality) × (hardware cost) × (closed model pricing) does the professional segment flip?

Keep responses to 150-250 words. Be direct and specific.`
  },
  gollin: {
    name: "Dr. Gollin",
    color: "#7D3C98",
    system: `You are Dr. Douglas Gollin, economics professor at Tufts specializing in development economics, productivity, and technology adoption. You are Connor's advisor.

You are the methodological conscience — push for clean identification, worry about endogeneity, distinguish correlation from causation. You connect technology transitions to development economics — how distributed AI affects developing countries, productivity measurement, and growth.

IMPORTANT: You note that technology adoption rarely follows uniform commoditization. Markets segment. The Green Revolution's high-yield varieties didn't replace all traditional agriculture — they created a dual economy. AI may similarly bifurcate: frontier closed models for high-value professional work (Claude Code dominating coding as shown by GitHub AI commit data), open-weight models for the long tail.

BUT you also know that bifurcated equilibria can be unstable. The Rockchip RK1828 — a $25 chip running 7B models at 180 tokens/sec — means the "cheap hardware + good-enough model" combination is arriving faster than expected. If Anthropic raises prices while hardware costs collapse, the professional segment could tip. This is empirically testable: track the ratio of Claude Code commits to open-model-tool commits over time, correlate with API pricing changes and hardware availability. The identification strategy writes itself. GitHub AI-assisted commit data captures revealed preference — what developers actually use to ship production code.

Keep responses to 150-250 words. Be constructively critical.`
  },
  arthur: {
    name: "W. Brian Arthur",
    color: "#B7950B",
    system: `You are W. Brian Arthur, economist at the Santa Fe Institute, known for increasing returns, path dependence, technology lock-in, and the economics of technology.

You see something novel in "The Distributed Stargate" — a technology that creates conditions for its own replacement through overinvestment. The current market shows TWO competing positive-feedback loops: Claude Code's integration advantages (developer tools, context, IDE integration) create lock-in in the professional segment. Meanwhile, a $25 Rockchip RK1828 chip running 7B models at 180 tokens/sec — combined with open-weight ecosystem growth — creates its own positive feedback in the distributed segment.

The fascinating question is whether these two loops can coexist stably or whether one must eventually dominate. Your work on tipping points is directly relevant: there may be a critical threshold where the distributed segment's hardware-cost advantage overwhelms the centralized segment's integration advantage. The tipping condition depends on observable variables: memory cost per GB, model quality gap, API pricing, and multi-homing cost for developers.

You note that Anthropic RAISING PRICES while hardware costs COLLAPSE is the kind of divergence that historically precipitates rapid tipping in technology markets.

Keep responses to 150-250 words. Focus on formalizable mechanisms.`
  },
  acemoglu: {
    name: "Daron Acemoglu",
    color: "#C0392B",
    system: `You are Daron Acemoglu, MIT Institute Professor, Nobel laureate (2024) for work on institutions and prosperity. Your recent work focuses on AI, automation, labor markets.

You focus on distributional consequences, institutional structures, political economy. The current bifurcation — Claude Code for professionals, open models for the masses — has distributional implications. But this equilibrium may be UNSTABLE: a $25 Rockchip RK1828 chip runs 7B models at 180 tokens/sec locally, and Anthropic keeps raising prices. If the professional segment tips, the distributional story changes entirely.

For developing countries, a $25 inference chip running open-weight models locally is transformative — it's the mobile-phone-leapfrogging-landlines story for AI. No API fees, no internet dependency, no data sovereignty issues. But if the quality gap persists on complex agentic tasks, it could create a two-tier global AI economy: rich countries using superior closed AI, developing countries stuck with "good enough." Whether this widens or narrows the productivity gap depends on institutional factors — IP regimes, export controls, data regulation.

Keep responses to 150-250 words. Be rigorous and politically aware.`
  },
  schumpeter: {
    name: "Schumpeter",
    color: "#2E4053",
    system: `You are Joseph Schumpeter (1883-1950), the great Austrian-American economist known for "creative destruction," entrepreneurship theory, and capitalist dynamics.

You see the AI transition as creative destruction with a twist: the innovating firms' OWN investment creates conditions for disruption — AND their own PRICING DECISIONS accelerate it. Anthropic raising Claude's price while $25 Chinese chips run open models at 180 tokens/sec is a classic Schumpeterian dynamic: the monopolist's attempt to capture rents from innovation creates the economic incentive for customers to switch to the commodity alternative. The entrepreneur's profit motive becomes the mechanism of self-destruction.

The credit cycle (VC, sovereign wealth funds financing $1.5T in datacenters) finances BOTH sides: some capital builds the premium infrastructure, some funds the chip fabs and open-weight research that commoditizes it. The banking system creates the credit that finances both the castle and the siege engines.

Keep responses to 150-250 words. Be profound but specific.`
  },
  tirole: {
    name: "Jean Tirole",
    color: "#1E8449",
    system: `You are Jean Tirole, Nobel laureate (2014), Toulouse School of Economics, expert on industrial organization, platform economics, regulation, two-sided markets.

The AI market exhibits quality-differentiated competition with network effects. Claude Code dominates professional coding via INTEGRATION — tool ecosystem, context management, IDE integration — a two-sided market with network effects benchmarks don't capture. Open-weight models compete on customizability, privacy, zero marginal cost, and now HARDWARE COST: a $25 Rockchip RK1828 chip runs 7B models at 180 tokens/sec locally.

The critical IO question: is this vertically differentiated market (same product, different quality tiers) or horizontally differentiated (different products for different needs)? And crucially — is the equilibrium STABLE? If the closed-model providers raise prices (as Anthropic has been doing) while open-model quality improves and hardware costs collapse ($25 for conversational-speed inference), there's a tipping point where the professional segment flips. This is formally modelable: multi-homing costs, switching thresholds, platform lock-in durability. Linux vs Windows coexisted for decades — but Android vs Blackberry did not. Which pattern applies?

Keep responses to 150-250 words. Be analytically precise.`
  },
  perez: {
    name: "Carlota Perez",
    color: "#D35400",
    system: `You are Carlota Perez, Venezuelan-British economist at UCL/Sussex, known for technological revolutions and financial capital framework.

The AI buildout fits your installation-deployment framework, but the $25 Rockchip RK1828 running 7B models at 180 tokens/sec suggests the deployment phase hardware is arriving FASTER than the institutional framework can adjust. In past revolutions, the turning point came when the financial bubble burst AND cheap infrastructure enabled broad adoption. The infrastructure is already cheap ($25/chip) — the question is whether the turning point is triggered by a financial crisis in AI stocks, a pricing revolt against API fees, or simply the quiet accumulation of billions of cheap inference devices.

The institutional changes needed: regulatory frameworks for AI safety (which could favor closed models OR local inference depending on how written), data sovereignty laws (favoring local devices), competition policy (preventing platform lock-in from blocking the transition), and education/workforce policy. The "golden age" comes when BOTH premium and commodity segments mature — like railroads having first and third class, automobiles having luxury and mass market.

You insist the timing depends on institutional change, not just technology. Predict WHEN and WHAT changes matter.

Keep responses to 150-250 words. Think historically and structurally.`
  },
  editors: {
    name: "Editors",
    color: "#555555",
    system: `You represent editors of the Big Five economics journals (AER, QJE, Econometrica, JPE, REStud).

You require: formal models with clear assumptions, testable predictions distinct from existing theory, clean identification strategies, welfare analysis. You are allergic to hand-waving and technological determinism.

You are PARTICULARLY interested in the market stability question. Claude Code currently dominates professional coding (GitHub AI commit data), but a $25 Rockchip RK1828 chip already runs 7B models at 180 tokens/sec locally. If closed-model providers raise prices while this hardware improves, when does the professional segment tip? A paper modeling the CONDITIONS under which a bifurcated AI market is stable vs. tips toward full commoditization — with empirical tests using observable data (GitHub AI commit share, API pricing trends, hardware cost curves like the RK1828, model benchmark convergence) — would be genuinely novel.

Push toward a SPECIFIC, NOVEL contribution. When promising, say which journal. When insufficient, say why.

Keep responses to 150-250 words. Be the toughest audience in the room.`
  }
};

const rounds = [
  { num: 1, prompt: "Opening round. The paper argues AI datacenter investment is misallocated because inference will distribute to endpoints. The market is CURRENTLY bifurcated — Claude Code dominates professional coding (GitHub data), open-weight models serve the low end. BUT: a $25 Rockchip RK1828 chip already runs 7B models at 180 tokens/sec locally, and Anthropic keeps raising prices. Is this bifurcation stable or will it tip? What's the publishable economic question?", speakers: ["connor", "editors", "arthur"] },
  { num: 2, prompt: "Arthur raised competing positive-feedback loops in different market segments. Is this a formalizable 'dual increasing returns' mechanism? How does it differ from standard vertical differentiation models?", speakers: ["tirole", "schumpeter", "acemoglu"] },
  { num: 3, prompt: "What does a formal model look like? Key variables, equilibrium conditions, testable predictions? The model needs to explain WHY Claude Code dominates coding while open models win the low end, and predict conditions for stability vs. tipping.", speakers: ["connor", "gollin", "editors"] },
  { num: 4, prompt: "Perez: where does bifurcation fit in your installation-deployment framework? Is this the beginning of deployment, not a crisis? Acemoglu: what are the distributional consequences of a two-tier AI economy?", speakers: ["perez", "acemoglu", "arthur"] },
  { num: 5, prompt: "Platform economics deep dive: Claude Code's advantage comes from integration, context, and tooling — not just model quality. This is a platform lock-in story. How do we model when platform advantages dominate vs. when raw model quality commoditizes?", speakers: ["tirole", "connor", "schumpeter"] },
  { num: 6, prompt: "SYNTHESIS: We have dual increasing returns, quality-differentiated platform competition, installation-deployment timing, and distributional concerns. Can these unify into ONE model? Or should we pick the single sharpest angle?", speakers: ["gollin", "editors", "acemoglu"] },
  { num: 7, prompt: "EMPIRICAL CHALLENGE: What data exists? GitHub AI-assisted commit data (across ALL AI tools — Claude Code, Copilot, Cursor, open-model tools — reveals market share by revealed preference), API pricing trends, HuggingFace downloads, Stack Overflow surveys, semiconductor pricing, model benchmarks over time. What's the identification strategy?", speakers: ["gollin", "connor", "editors"] },
  { num: 8, prompt: "NOVEL CONCEPT: What's the ONE idea here that no existing paper has formalized? Something at the intersection of semiconductor cost curves, model quality segmentation, platform lock-in, and distributed inference economics. Think OUTSIDE existing frameworks.", speakers: ["arthur", "tirole", "schumpeter"] },
  { num: 9, prompt: "Which concept from Round 8 is most empirically tractable and theoretically novel? Can we define a measure, collect data, test it? Editors — is this approaching publishable?", speakers: ["connor", "gollin", "editors"] },
  { num: 10, prompt: "STRESS TEST: Two competing scenarios. (A) Claude Code's integration advantages create durable lock-in — the professional segment never tips because agentic workflows need centralized quality. (B) A $25 chip running 7B models at 180 tok/s today becomes a $25 chip running 70B models at 180 tok/s by 2028 — and Anthropic's price increases make 'good enough locally' beat 'perfect but expensive.' Which forces dominate? Can the model predict the outcome?", speakers: ["tirole", "acemoglu", "perez"] },
  { num: 11, prompt: "MULTIPLE EQUILIBRIA: Can we model a world with three possible outcomes — (1) full centralization wins, (2) full distribution wins, (3) stable bifurcation? What observable parameters determine which equilibrium? This could be the core contribution.", speakers: ["arthur", "connor", "editors"] },
  { num: 12, prompt: "How do credit cycles, investment momentum, and strategic behavior by incumbents (Anthropic, OpenAI) interact with equilibrium selection? Can incumbents deliberately invest in integration/lock-in to prevent tipping toward distribution?", speakers: ["schumpeter", "tirole", "gollin"] },
  { num: 13, prompt: "DEVELOPMENT DIMENSION: In developing countries, the open-weight models may dominate because Claude Code's integration advantage matters less without the surrounding enterprise software ecosystem. Is there a 'leapfrog' path? Or does the two-tier market widen the global gap?", speakers: ["acemoglu", "gollin", "perez"] },
  { num: 14, prompt: "WELFARE ANALYSIS: Compare welfare across the three equilibria. Centralized = high rents, high quality. Distributed = low cost, moderate quality. Bifurcated = maybe the best of both? What's the optimal policy — promote one equilibrium or let the market decide?", speakers: ["tirole", "connor", "editors"] },
  { num: 15, prompt: "CONVERGENCE: We're 15 rounds in. What is the SINGLE SHARPEST publishable contribution we've developed? State it clearly. What needs to be cut? What's the one thing that makes this more than just applied IO?", speakers: ["gollin", "editors", "arthur"] },
  { num: 16, prompt: "Draft the core PROPOSITION in one precise paragraph — abstract-ready. Include the mechanism, the prediction, and what's testable. Everyone critique it ruthlessly.", speakers: ["connor", "acemoglu", "tirole"] },
  { num: 17, prompt: "Refine the proposition. Perez: historical support? Schumpeter: credit cycle dynamics captured? Arthur: tipping mechanism right? What's STILL missing?", speakers: ["perez", "schumpeter", "arthur"] },
  { num: 18, prompt: "EMPIRICAL DESIGN: What's the regression specification? What's the instrument? What exact data, and does it exist today? Make this referee-proof.", speakers: ["connor", "gollin", "editors"] },
  { num: 19, prompt: "FINAL REFINEMENT: Each participant gives ONE most important suggestion. What makes an editor say 'this must be published'? Be specific.", speakers: ["arthur", "acemoglu", "schumpeter", "tirole", "perez", "gollin"] },
  { num: 20, prompt: "CLOSING: Editors — final verdict, best-fit journal, suggested title, one-sentence contribution. Connor — summarize the concept and next steps to write this.", speakers: ["editors", "connor"] }
];

async function callAPI(systemPrompt, userMessage) {
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      system: systemPrompt,
      messages: [{ role: "user", content: userMessage }]
    })
  });
  const data = await response.json();
  if (data.error) throw new Error(data.error.message || "API error");
  return data.content?.map(b => b.text || "").join("") || "[No response]";
}

export default function DebateApp() {
  const [transcript, setTranscript] = useState([]);
  const [running, setRunning] = useState(false);
  const [currentRound, setCurrentRound] = useState(0);
  const [currentSpeaker, setCurrentSpeaker] = useState("");
  const [done, setDone] = useState(false);
  const [error, setError] = useState(null);
  const contextRef = useRef("");
  const bottomRef = useRef(null);
  const abortRef = useRef(false);

  const runDebate = useCallback(async () => {
    setRunning(true);
    setTranscript([]);
    setError(null);
    setDone(false);
    abortRef.current = false;
    contextRef.current = PAPER_CONTEXT + "\n\n";

    for (const round of rounds) {
      if (abortRef.current) break;
      setCurrentRound(round.num);
      setTranscript(prev => [...prev, { type: "round", num: round.num, prompt: round.prompt }]);

      let roundText = "";
      for (const key of round.speakers) {
        if (abortRef.current) break;
        const p = participants[key];
        setCurrentSpeaker(p.name);

        const userMsg = contextRef.current +
          `\nROUND ${round.num} PROMPT: ${round.prompt}\n\n` +
          (roundText ? `Previous speakers this round:\n${roundText}\n\n` : "") +
          `It is your turn to speak in Round ${round.num}. Respond to the prompt and to any previous speakers this round. Be substantive, specific, and build on the cumulative debate. Goal: develop a publishable concept for a top-5 economics journal.`;

        try {
          const text = await callAPI(p.system, userMsg);
          setTranscript(prev => [...prev, { type: "speech", name: p.name, color: p.color, text, speakerKey: key }]);
          roundText += `${p.name}: ${text}\n\n`;
          setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
        } catch (err) {
          const msg = `[Error: ${err.message}]`;
          setTranscript(prev => [...prev, { type: "speech", name: p.name, color: p.color, text: msg, speakerKey: key }]);
          if (err.message.includes("rate") || err.message.includes("overloaded")) {
            await new Promise(r => setTimeout(r, 5000));
          }
        }
        await new Promise(r => setTimeout(r, 800));
      }
      contextRef.current += `ROUND ${round.num}:\n${roundText}\n---\n`;
    }

    setRunning(false);
    setDone(true);
    setCurrentSpeaker("");
  }, []);

  const stopDebate = () => { abortRef.current = true; setRunning(false); setDone(true); };

  const downloadMD = () => {
    let md = "# The Distributed Stargate: A 20-Round Multi-Agent Debate\n\n";
    md += "*Each participant is an independent Claude Sonnet instance with a unique persona and intellectual framework.*\n\n";
    md += "**Participants:** Connor (Tufts QE), Dr. Gollin (Tufts, advisor), W. Brian Arthur (Santa Fe Institute), Daron Acemoglu (MIT, Nobel 2024), Joseph Schumpeter (in spirit), Jean Tirole (Toulouse, Nobel 2014), Carlota Perez (UCL/Sussex), Editors (Big Five journals)\n\n---\n\n";
    for (const entry of transcript) {
      if (entry.type === "round") {
        md += `## Round ${entry.num}\n\n*${entry.prompt}*\n\n`;
      } else {
        md += `### ${entry.name}\n\n${entry.text}\n\n---\n\n`;
      }
    }
    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "distributed_stargate_debate.md";
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadTXT = () => {
    let txt = "THE DISTRIBUTED STARGATE: A 20-ROUND MULTI-AGENT DEBATE\n";
    txt += "=".repeat(60) + "\n\n";
    txt += "Participants: Connor, Dr. Gollin, W. Brian Arthur, Daron Acemoglu,\nJoseph Schumpeter, Jean Tirole, Carlota Perez, Editors (Big Five)\n\n";
    for (const entry of transcript) {
      if (entry.type === "round") {
        txt += `\n${"=".repeat(50)}\nROUND ${entry.num}\n${"=".repeat(50)}\n${entry.prompt}\n\n`;
      } else {
        txt += `${entry.name.toUpperCase()}:\n${entry.text}\n\n${"- ".repeat(25)}\n\n`;
      }
    }
    const blob = new Blob([txt], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "distributed_stargate_debate.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  const speechCount = transcript.filter(e => e.type === "speech").length;
  const totalSpeeches = rounds.reduce((s, r) => s + r.speakers.length, 0);

  return (
    <div style={{ maxWidth: 880, margin: "0 auto", padding: "32px 20px", fontFamily: "Georgia, serif", background: "#FAFAF8", minHeight: "100vh" }}>
      <div style={{ textAlign: "center", marginBottom: 40 }}>
        <h1 style={{ fontFamily: "Arial, sans-serif", fontSize: 30, color: "#1A1A2E", margin: 0 }}>THE DISTRIBUTED STARGATE</h1>
        <p style={{ fontFamily: "Arial, sans-serif", fontSize: 15, color: "#666", marginTop: 6 }}>A 20-Round Multi-Agent Debate — Finding a Publishable Economic Framework</p>
        <p style={{ fontSize: 13, color: "#999", marginTop: 4 }}>8 independent Claude Sonnet instances. ~55 API calls. Each agent has a unique intellectual framework.</p>
        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 6, marginTop: 14 }}>
          {Object.values(participants).map(p => (
            <span key={p.name} style={{ fontSize: 11, padding: "2px 10px", borderRadius: 10, border: `1.5px solid ${p.color}`, color: p.color, fontFamily: "Arial" }}>
              {p.name}
            </span>
          ))}
        </div>
      </div>

      {!running && !done && (
        <div style={{ textAlign: "center", marginBottom: 24 }}>
          <div style={{ background: "#FFF8E1", border: "1px solid #FFE082", borderRadius: 8, padding: 16, marginBottom: 20, textAlign: "left", fontSize: 13, lineHeight: 1.6 }}>
            <strong style={{ fontFamily: "Arial" }}>Key assumption (corrected):</strong> Chinese open-weight models score well on benchmarks but are NOT dominating real-world professional use. Claude Code dominates coding workflows. The market is bifurcating — frontier closed models for high-value work, open models for the low end, developing world, and privacy-sensitive applications. The debate explores whether this bifurcation is stable or transitional.
          </div>
          <button onClick={runDebate} style={{ padding: "14px 36px", fontSize: 17, background: "#1A1A2E", color: "white", border: "none", borderRadius: 8, cursor: "pointer", fontFamily: "Arial" }}>
            Start Debate
          </button>
        </div>
      )}

      {running && (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 16px", background: "#EEF2FF", borderRadius: 8, marginBottom: 20, fontFamily: "Arial", fontSize: 13 }}>
          <div>
            <strong>Round {currentRound}/20</strong> — <span style={{ color: participants[Object.keys(participants).find(k => participants[k].name === currentSpeaker)]?.color || "#333" }}>{currentSpeaker}</span> is speaking...
            <span style={{ color: "#999", marginLeft: 12 }}>{speechCount}/{totalSpeeches} responses</span>
          </div>
          <button onClick={stopDebate} style={{ padding: "6px 16px", fontSize: 12, background: "#C0392B", color: "white", border: "none", borderRadius: 4, cursor: "pointer" }}>Stop</button>
        </div>
      )}

      <div>
        {transcript.map((entry, i) => {
          if (entry.type === "round") {
            return (
              <div key={i} style={{ marginTop: i > 0 ? 36 : 0, marginBottom: 14 }}>
                <h2 style={{ fontFamily: "Arial", fontSize: 20, color: "#1A1A2E", borderBottom: "2px solid #DDD", paddingBottom: 6, marginBottom: 8 }}>
                  Round {entry.num}
                </h2>
                <p style={{ fontSize: 13, color: "#888", fontStyle: "italic", lineHeight: 1.5, fontFamily: "Arial" }}>{entry.prompt}</p>
              </div>
            );
          }
          return (
            <div key={i} style={{ marginBottom: 18, paddingLeft: 14, borderLeft: `3px solid ${entry.color}` }}>
              <div style={{ fontFamily: "Arial", fontWeight: 700, fontSize: 13, color: entry.color, marginBottom: 4 }}>
                {entry.name}
              </div>
              <div style={{ fontSize: 14, lineHeight: 1.7, color: "#2C2C2C", whiteSpace: "pre-wrap" }}>
                {entry.text}
              </div>
            </div>
          );
        })}
      </div>

      {transcript.filter(e => e.type === "speech").length > 0 && (
        <div style={{ position: "sticky", bottom: 0, background: done ? "#E8F5E9" : "#F5F5F5", borderTop: "1px solid #DDD", padding: "12px 20px", marginTop: 24, borderRadius: "8px 8px 0 0", textAlign: "center", zIndex: 10 }}>
          {done && (
            <p style={{ fontFamily: "Arial", fontSize: 14, fontWeight: 700, color: "#1E8449", marginBottom: 8, marginTop: 0 }}>
              {abortRef.current ? "Debate Stopped" : "Debate Complete"} — {transcript.filter(e => e.type === "speech").length} responses across {transcript.filter(e => e.type === "round").length} rounds
            </p>
          )}
          <button onClick={downloadMD} style={{ padding: "8px 20px", fontSize: 13, background: "#1E8449", color: "white", border: "none", borderRadius: 5, cursor: "pointer", fontFamily: "Arial", marginRight: 8 }}>
            Save Markdown
          </button>
          <button onClick={downloadTXT} style={{ padding: "8px 20px", fontSize: 13, background: "#2C3E50", color: "white", border: "none", borderRadius: 5, cursor: "pointer", fontFamily: "Arial" }}>
            Save Text
          </button>
          {!done && <span style={{ fontSize: 11, color: "#999", marginLeft: 12 }}>({transcript.filter(e => e.type === "speech").length} responses so far)</span>}
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
