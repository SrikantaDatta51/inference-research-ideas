#!/usr/bin/env python3
"""Generate all 3 deep papers at full length (14, 12, 14 pages)."""
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def write_file(subdir, name, content):
    path = os.path.join(BASE, subdir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    lines = content.count('\n')
    print(f"  Written {subdir}/{name} ({lines} lines)")

# ═══════════════════════════════════════════════════════════════
# PAPER 5: InferTwin — 14 pages (~1400 lines)
# ═══════════════════════════════════════════════════════════════
print("=== Paper 5: InferTwin (14 pages) ===")

p5_bib = r"""@article{hafner2023dreamerv3,title={Mastering Diverse Domains through World Models},author={Hafner, D. and Pasukonis, J. and Ba, J. and Lillicrap, T.},journal={arXiv:2301.04104},year={2023}}
@article{chen2018neural,title={Neural Ordinary Differential Equations},author={Chen, R.T.Q. and Rubanova, Y. and Bettencourt, J. and Duvenaud, D.},journal={NeurIPS},year={2018}}
@article{li2020scalable,title={Scalable Gradients for Stochastic Differential Equations},author={Li, X. and Wong, T. and Chen, R.T.Q. and Duvenaud, D.},journal={AISTATS},year={2020}}
@article{pearl2009causality,title={Causality: Models, Reasoning and Inference},author={Pearl, J.},publisher={Cambridge Univ. Press},year={2009}}
@article{kumar2020cql,title={Conservative Q-Learning for Offline RL},author={Kumar, A. and Zhou, A. and Tucker, G. and Levine, S.},journal={NeurIPS},year={2020}}
@article{schulman2017ppo,title={Proximal Policy Optimization Algorithms},author={Schulman, J. and Wolski, F. and Dhariwal, P. and Radford, A. and Klimov, O.},journal={arXiv:1707.06347},year={2017}}
@article{nvidia2025dcgm,title={NVIDIA Data Center GPU Manager},author={{NVIDIA Corporation}},year={2025}}
@article{kwon2023vllm,title={Efficient Memory Management for LLM Serving with PagedAttention},author={Kwon, W. and Li, Z. and Zhuang, S. and others},journal={SOSP},year={2023}}
@article{grieves2014digital,title={Digital Twin: Manufacturing Excellence through Virtual Factory Replication},author={Grieves, M. and Vickers, J.},year={2014}}
@article{tao2018digital,title={Digital Twin in Industry: State-of-the-Art},author={Tao, F. and Sui, F. and Liu, A. and others},journal={IEEE TII},volume={15},number={4},year={2018}}
@article{mao2019learning,title={Learning Scheduling Algorithms for Data Processing Clusters},author={Mao, H. and Schwarzkopf, M. and others},journal={SIGCOMM},year={2019}}
@article{mirhoseini2021graph,title={A Graph Placement Methodology for Fast Chip Design},author={Mirhoseini, A. and Goldie, A. and others},journal={Nature},volume={594},year={2021}}
@article{chen2021decision,title={Decision Transformer: RL via Sequence Modeling},author={Chen, L. and Lu, K. and Rajeswaran, A. and others},journal={NeurIPS},year={2021}}
@article{schrittwieser2020muzero,title={Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model},author={Schrittwieser, J. and others},journal={Nature},volume={588},year={2020}}
@article{haarnoja2018sac,title={Soft Actor-Critic},author={Haarnoja, T. and Zhou, A. and Abbeel, P. and Levine, S.},journal={ICML},year={2018}}
@article{tessler2018rcpo,title={Reward Constrained Policy Optimization},author={Tessler, C. and Mankowitz, D. and Mannor, S.},journal={ICLR},year={2018}}
@article{patel2024splitwise,title={Splitwise: Efficient Generative LLM Inference Using Phase Splitting},author={Patel, P. and others},journal={ISCA},year={2024}}
@article{nvidia2025blackwell,title={NVIDIA Blackwell Architecture},author={{NVIDIA Corporation}},year={2025}}
@article{vezhnevets2017feudal,title={Feudal Networks for Hierarchical RL},author={Vezhnevets, A. and others},journal={ICML},year={2017}}
@article{sutton1999options,title={Between MDPs and Semi-MDPs: Temporal Abstraction in RL},author={Sutton, R. and Precup, D. and Singh, S.},journal={Artificial Intelligence},volume={112},year={1999}}
@article{kiran2022digital,title={Digital Twins for Data Center Infrastructure Management},author={Kiran, M. and others},journal={IEEE Cloud},year={2022}}
@article{tobin2017domain,title={Domain Randomization for Sim-to-Real Transfer},author={Tobin, J. and others},journal={IROS},year={2017}}
@article{achiam2017cpo,title={Constrained Policy Optimization},author={Achiam, J. and Held, D. and Tamar, A. and Abbeel, P.},journal={ICML},year={2017}}
@article{kingma2014vae,title={Auto-Encoding Variational Bayes},author={Kingma, D. and Welling, M.},journal={ICLR},year={2014}}
@article{levine2020offline,title={Offline RL: Tutorial, Review, and Perspectives},author={Levine, S. and Kumar, A. and Tucker, G. and Fu, J.},journal={arXiv:2005.01643},year={2020}}
@article{zheng2024sglang,title={SGLang: Efficient Execution of Structured LM Programs},author={Zheng, L. and others},journal={arXiv:2312.07104},year={2024}}
@article{iea2025datacenter,title={Data Centres and AI Energy Consumption Outlook},author={{International Energy Agency}},year={2025}}
@article{sun2024cxl,title={Demystifying CXL Memory with Genuine Devices},author={Sun, Y. and others},journal={ASPLOS},year={2024}}
@article{patterson2022carbon,title={Carbon Emissions and Large Neural Network Training},author={Patterson, D. and others},journal={arXiv:2104.10350},year={2022}}
@article{kolesnikov2024survey,title={RL for Cloud Resource Management: A Survey},author={Kolesnikov, S. and others},journal={ACM Computing Surveys},year={2024}}
@article{rashid2020qmix,title={Monotonic Value Function Factorisation for Deep Multi-Agent RL},author={Rashid, T. and others},journal={JMLR},year={2020}}
@article{ha2018world,title={World Models},author={Ha, D. and Schmidhuber, J.},journal={arXiv:1803.10122},year={2018}}
@article{lecarpentier2021non,title={Non-Stationary MDPs},author={Lecarpentier, E. and Rachelson, E.},journal={NeurIPS},year={2021}}
@article{koller2009pgm,title={Probabilistic Graphical Models},author={Koller, D. and Friedman, N.},publisher={MIT Press},year={2009}}
"""
write_file("paper5-rl-world-model-digital-twin", "references.bib", p5_bib)

print("  Generating main.tex...")
