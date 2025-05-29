The Weaver’s Code: ArGen and the
Auto-Regulation of Generative AI
Kapil Madan, Principled Evolution
kmadan@principledevolution.ai
Abstract
This paper introduces ArGen (Auto-Regulation of Generative AI systems),
a novel framework to align large language models (LLMs) with multifaceted
human values and configurable ethical principles. ArGen employs a robust
strategy that integrates automated reward function generation, Group Rel-
ative Policy Optimisation (GRPO), and Open Policy Agent (OPA). This
approach allows for the formal encoding of diverse ethical considerations
and constraints, guiding reinforcement learning towards safer and more
aligned AI behaviours. We detail a Python-based implementation where
programmable reward functions and OPA policies enable fine-grained con-
trol over agent conduct. The framework is designed to be adaptable to
various value systems; as a specific introduction and case study, we explore
its application in the development of an AI assistant guided by principles
derived from Dharmic ethics, focussing on safety (Ahimsa), scope adher-
ence (Dharma), and helpfulness. This demonstrates ArGen’s capacity for
culturally nuanced alignment. Our work draws upon established AI safety
literature (e.g., RLHF, Constitutional AI) and illustrates the integration
of external policy engines for dynamic AI governance. We present the ar-
chitecture, including the policy integration and decision-making workflows.
Through the development of this demonstration repository, we show that
ArGen’s methodology can address key alignment challenges such as value
specification, situational appropriateness, and fostering trust. This research
lays a foundation for policy-driven, auto-adaptive AI alignment, offering a
pathway toward LLMs that are technically proficient, ethically robust, and
adaptable for safe deployment in diverse global contexts.
Keywords: Custom AI Models; AI Regulation; Group Relative Pol-
icy Optimisation; AI alignment; Dharmic ethics; Bhagavad Gita;
Open Policy Agent; Reinforcement Learning; AI safety; ethical
AI
1 Introduction
The rapid advancement of Large Language Models (LLMs) presents both transformative
opportunities and significant societal challenges. Ensuring these powerful generative AI sys-
tems operate safely, beneficially, and in accordance with diverse human values - a pursuit
broadly termed AI alignment - has become a critical research imperative [cite general AI
safety overview, e.g., Hendrycks et al. on measuring progress]. Although current alignment
techniques, such as Reinforcement Learning from Human Feedback (RLHF) [Ouyang et
al., 2022] and Constitutional AI [Bai et al., 2022], have made strides in improving helpful-
ness and reducing overt harms, the task of imbuing LLMs with nuanced, adaptable, and
auditable ethical conduct remains complex. Existing methods can be resource intensive,
37th Conference on Neural Information Processing Systems (NeurIPS 2023).
may inadvertently encode a narrow range of values, and often lack mechanisms for dynamic
governance in response to evolving societal norms or contextual requirements.
This paper introduces ArGen (Auto-Regulation of Generative AI systems), a novel frame-
work designed to address these challenges by enabling a more continuous, configurable and
policy-driven approach to LLM alignment. ArGen conceptualises alignment not as a static
endpoint, but as an ongoing process of autoregulation, where an AI system’s behaviours
are shaped by a dynamic interplay of programmable reward functions, robust reinforcement
learning, and an explicit governance layer. At its core, ArGen functions as a ”weaver’s code,”
providing the machinery to intricately interlace diverse ethical principles and operational
policies into the fabric of an LLM’s decision-making processes.
The ArGen framework integrates three key technical pillars:
1. Automated Reward Function Generation: Leveraging capable LLMs as eval-
uators (LLM-as-a-Judge) to translate configurable ethical principles and desired be-
haviours into granular reward signals. This allows for the creation of multifaceted
reward functions that can be adapted to different value systems.
2. Group Relative Policy Optimisation (GRPO): Using an advanced reinforce-
ment learning algorithm designed for stable and efficient policy updates, allowing
the LLM to learn from the generated complex reward landscape.
3. Open Policy Agent (OPA) Based Governance: Integrating an external OPA
policy engine to enforce formally defined constraints and ethical rules, allowing
for dynamic updates and providing an auditable layer of control over the LLM’s
conduct.
We detail a Python-based implementation of ArGen, demonstrating how abstract principles
can be operationalised into programmable reward functions and OPA policies. To showcase
ArGen’s adaptability and its capacity for culturally nuanced alignment, we present an in-
depth case study: the development of ”MedGuide-AI,” a medical information assistant. This
instantiation of ArGen is guided by key principles derived from Dharmic ethics—specifically
Ahimsa (non-harm/safety), Dharma (adherence to the appropriate scope and duty) and
holistic helpfulness (encompassing clarity, completeness, relevance, and empathy). This case
study serves not to exclusively advocate for one ethical system, but to illustrate ArGen’s
broader capability to incorporate diverse and specific value sets.
Our primary contributions are as follows.
• The conceptualisation and implemented design of the ArGen framework, offering a
novel synthesis of automated reward generation, GRPO, and OPA for the autoreg-
ulation of generative AI.
• A methodology for translating ethical principles, including those from culturally
specific contexts like Dharmic ethics, into concrete, machine-interpretable reward
signals and OPA policies within the ArGen architecture.
• A demonstration, through our open source repository and case study, of ArGen’s
feasibility in improving LLM alignment along multiple ethical dimensions and its
potential to address challenges in value specification and situational appropriate-
ness.
This research draws upon established AI safety literature and aims to advance the develop-
ment of policy-driven, auto-adaptive AI alignment. We argue that frameworks like ArGen
offer a pathway towards LLMs that are not only technically proficient but also ethically
robust and adaptable for responsible deployment in a diverse global landscape. The remain-
der of this paper details ArGen’s architecture, its implementation, the specifics of the case
study, and discusses its broader implications and avenues for future work.
2 Related Work
Aligning advanced Artificial Intelligence (AI) with multifaceted human values and inten-
tions is a paramount challenge in contemporary AI research [Russell, 2019; Hendrycks et al.,
2
2023 - annotation: Add a recent general overview of AI safety / alignment if Hendrycks is
more focused on benchmarks]. The potential for AI systems to optimise misspecified or in-
complete objectives, leading to undesirable outcomes (as illustrated by thought experiments
such as the paperclip maximiser [Bostrom, 2003]), underscores the critical need for robust
mechanisms to instil ethical considerations and human-compatible goals into AI systems.
Our framework, ArGen, synthesises and extends several key research threads to address
this challenge.
2.1 Reinforcement Learning for AI Alignment
Reinforcement Learning from Human Feedback (RLHF) has become a cornerstone tech-
nique for steering large language models (LLM) towards desired behaviours [Ouyang et al.,
2022; Christiano et al., 2017]. Approaches such as InstructGPT [Ouyang et al., 2022] demon-
strated that reward models trained on human preference data can significantly enhance LLM
helpfulness and harmlessness. Building on this, Constitutional AI (CAI) [Bai et al., 2022]
introduced a method for alignment using a predefined set of principles (a ’constitution’) to
guide AI-generated feedback (Reinforcement Learning from AI Feedback - RLAIF), thereby
reducing direct human labelling effort for adherence to these principles. These methods
highlight the efficacy of using RL to internalise complex behavioural preferences.
ArGen utilises Group Relative Policy Optimisation (GRPO) [DeepSeek-AI, 2024; Shao et al.,
2024], an advance over Proximal Policy Optimisation (PPO) [Schulman et al., 2017], which
has shown strong performance in optimising LLMs for complex reasoning tasks, sometimes
without requiring a separate value function. For instance, GRPO has been applied suc-
cessfully in mathematical reasoning [Shao et al., 2024] and code generation, such as in the
SWE-RL project, where rule-based rewards from patch comparisons effectively guided the
model [Wei et al., 2025 - annotation: The Wei et al. 2025 citation in your draft for
SWE-RL seems to be Weidinger 2021 on ethical models; double-check the correct citation for
SWE-RL if it used GRPO. If SWE-RL used a different RL algorithm but had rule-based re-
wards, clarify that, e.g., ”RL agents, as seen in X, can be effectively trained with...”]. ArGen
leverages GRPO’s stability and efficiency to learn from a multifaceted reward signal that
includes scores from automated principle evaluators and feedback from an external policy
engine.
Annotation for Deeper Research (RL-Based Alignment):
• Latest GRPO Applications/Variants: Search for any very recent (late 2024,
early 2025) applications, analyses or improvements of GRPO or similar policy opti-
misation algorithms for LLMs.
• Scalable Oversight & Reward Modelling: Briefly look at the latest on scal-
able oversight techniques beyond simple preference pairs for RLHF, and recent
advancements in reward modelling fidelity or LLM-as-a-judge capabilities, since Ar-
Gen relies on this for its automated rewards. Cite key papers (e.g., from Google,
Anthropic, OpenAI, or academic groups on these topics).
• Guidance: Focus on papers that discuss the stability, sample efficiency, or suitabil-
ity of RL algorithms for complex, multi-objective reward functions like those ArGen
might generate.
2.2 Policy-Based Governance and Control in AI Systems
Complementary to learning-based alignment, explicit rule-based governance provides mech-
anisms for enforcing hard constraints and codifying non-negotiable principles. Conceptually,
this dates back to ideas like Asimov’s Laws, but modern implementations offer greater so-
phistication. The Open Policy Agent (OPA) [Open Policy Agent, 2023] is a widely adopted
open-source policy engine to create unified, context-aware policy enforcement across various
software systems [Harness Developer Hub, n.d.; Spurin, 2021 - annotation: Add another
general OPA citation if available, perhaps from CNCF or a survey on policy engines]. OPA
uses a declarative language, Rego, to define policies that can be queried to make decisions
(e.g., allow/deny, data filtering).
3
Although OPA is prevalent in cloud native infrastructure and application authorisation, its
application as a dynamic externalised governance layer for LLM alignment during training
and inference is an emerging area. Some AI safety frameworks propose ”governor” modules
or external oversight mechanisms to intercept or guide AI actions [Saunders et al., 2022].
The Governance OPA Library (GOPAL) initiative further advocates for creating reusable
libraries of OPA policies specifically for AI systems [Principled-Evolution, 2025 - Annotation:
Ensure that this is the most appropriate citation for GOPAL, e.g., a whitepaper, repo, or
your own prior work if applicable]. ArGen distinctively integrates OPA directly into its
auto-regulatory loop, using OPA policies not only for potential pre/post filtering of LLM
interactions but also as a source of information for the reward function, thereby allowing
the RL agent to learn to act in accordance with formally specified rules. This provides a
transparent and dynamically updateable mechanism for encoding ethical guardrails.
Annotation for Deeper Research (Policy-Based Governance):
• OPA in AI/ML Contexts: Search specifically for existing (even if nascent) uses
or proposals of OPA or similar policy engines (e.g., Kyverno if relevant, though OPA
is more general) in the context of governing machine learning model behaviour, data
handling in ML pipelines or content generation.
• Formal Verification & AI Ethics: Briefly explore whether there are connections
to research on formal methods or symbolic AI used to verify or enforce ethical
constraints in AI systems, as OPA policies have a formal logical basis.
• Guidance: The aim is to show that while policy engines are established, their
specific, deep integration into an RL alignment loop like ArGen’s is novel.
2.3 Configurable Ethics and Culturally Aware AI Alignment
A significant challenge in AI alignment is the specification and integration of diverse human
values. Much of the foundational work on machine ethics and AI alignment has implicitly
or explicitly drawn from Western philosophical traditions (e.g., utilitarianism, deontology)
[Wallach & Allen, 2008]. However, there is a growing consensus on the need for AI systems
that are sensitive to a broader spectrum of cultural and ethical perspectives to ensure global
trust and equitable benefit [Mohamed et al., 2020; Sambasivan et al., 2021; Avin et al., 2021
- Annotation: Add Avin et al. on ”sociotechnical” perspective or similar broad papers on
value alignment challenges].
Varshney [2024] advocates the ”Decolonial AI Alignment,” advocating the incorporation
of concepts like viśeṣa-dharma (context-specific duties from Hindu philosophy) to create
more pluralistic and contextually aware AI moralities. Similarly, initiatives such as the
Susiddha AI Project explore Dharmic frameworks (including the puruṣārthas or human
aims) as a basis for AI goal systems that extend beyond narrow task optimisation [Susiddha
AI Project, n.d.]. Other research has explored AI alignment through Buddhist principles of
compassion [Feldman, 2019 - Annotation: Find a suitable citation for Buddhist ethics & AI
if desired] or other indigenous knowledge systems. These efforts highlight a common theme:
the potential for ancient wisdom traditions to offer rich, time-tested frameworks for guiding
AI development towards human flourishing.
ArGen is designed with configurability at its core, allowing different sets of principles to be
operationalised as reward functions and OPA policies. Our case study, detailed in Section 5,
instantiates ArGen using key principles from Dharmic ethics (such as Ahimsa – non-harm,
and Dharma – adherence to appropriate scope and duty) as derived from texts such as
the Bhagavad Gita. This specific application demonstrates ArGen’s capability to integrate
such culturally nuanced ethical considerations, aiming to produce AI behaviour that is not
only technically proficient but also ethically grounded within a chosen value system. This
approach seeks to mitigate biases that may arise from training data reflecting a limited set
of cultural values by allowing explicit encoding of diverse ethical priorities.
Annotation for Deeper Research (Cultural/Ethical Frameworks):
• Operationalizing Diverse Ethics: Look for more examples or methodologies
papers (if any exist) on how abstract ethical principles from various cultures (beyond
4
Dharmic, if you want to show breadth of the problem ArGen addresses) are being
proposed for operationalisation in AI.
• Critiques of Universalism in AI Ethics: Cite papers that specifically critique
”one-size-fits-all” or overly universalist approaches to AI ethics.
• Kids’ Use of AI/LLMs: If you decide this is relevant to your paper’s overall
message or a specific aspect of the case study (e.g., if MedGuide-AI policies implicitly
consider vulnerable users):
– Find 1-2 recent, impactful papers on ”AI safety for children,” ”ethical consid-
erations for LLMs used by minors,” or ”designing age-appropriate AI.”
– You could add a sentence like: ”Furthermore, the need for adaptable ethical
frameworks becomes even more acute when considering AI systems for vulnera-
ble populations, such as children, where specific safeguards and developmental
appropriateness are paramount [Cite Key Children & AI Ethics Paper 1, Paper
2].”
• Guidance: The goal is to position ArGen as a technical framework that can support
this move towards more inclusive and configurable AI ethics, with your case study
being one concrete example.