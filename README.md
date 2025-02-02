# Time-Dependent Quadratic Alignment Problem (TD-QAP)

Given NxN flow and distance matrices with time dependence, we implement two algorithms to solve the time-dependent quadratic alignment problem:
1. A classical computing, brute-force approach: Perfect accuracy is guaranteed but will only reliably finish for N <= 12. To show how quickly it scales, our data predicts the algorithm will take 70 minutes for N = 11, and the age of the universe for N = 21.
2. A quantum computing approach using D-Wave Systems' Quantum Annealers: Almost perfect accuracy (~99%) for tested N up to N = 50. Provides sufficient accuracy and precision with minimal quantum fluctuations in a fraction of the time (~65s for N = 20, ~42m for N = 50). Most of this time is spent calculating the costs (as we are using D-Wave's hybrid computers) and the QPU usage is kept under 3 seconds even when N = 50.

Our approach relies on converting QAP into a quadratic unconstrained binary optimization problem that can be modeled by the Ising Model. This ensures scalability if we wish to transition to fully quantum hardware. We compute the objective function by summing the costs and using it as our Hamiltonian energy function, which the quantum computer should minimize.

<img src="https://github.com/user-attachments/assets/48d3cf93-1b84-48a3-9d4a-8f425746d477" width="800px">
<br><br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp<img src="https://github.com/user-attachments/assets/d4344530-f4e6-48a1-bb52-5c8b16eb1114" width="650px">
<br><br>
<img src="https://cdn.discordapp.com/attachments/1306438341385781281/1335452121050513490/image.png?ex=67a03835&is=679ee6b5&hm=dad4c8aafd89d63d8adb1ebc2a1c4928b791ae07de90327395385d03eb57f584&" width="800px">

Example data mapping the TD-QAP problem to the seasonal allocation of flight/air train/bus allocation in airports based on popular destinations:

<img src="https://github.com/user-attachments/assets/0ee83cbd-b56a-4905-a0a1-ab27f32a0d8d" width="1200px">

The quantum algorithm has been shown to work with 99% accuracy for values of N < 10 when compared to the brute force algorithm, and its costs remain reasonable as N scales up. Thus, we can make comparisons with various slightly inaccurate classical heuristics, but since quantum computing still has a long way to go and this result was achieved with the number of qubits in the hundreds, the technology is extremely promising.
