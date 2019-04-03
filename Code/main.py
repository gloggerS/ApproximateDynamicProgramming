Seien $\pi = (\pi_1, ..., \pi_m)$ die Opportunitätskosten der $m$ Ressourcen. Dann ergibt sich als Approximation der Wertfunktion bei Kapazität $c \in \mathbb{N}_0^m$ und Zeit $t$
$$ V_t(c) \approx \hat{V}_t^i(c_i) + \sum_{k \neq i} \pi_k c_k$$

Als Opportunitätskosten für Produkt $j$ ergeben sich nun
\begin{align}
V_t(c) - V_t(c - A_j) & \approx \hat{V}_t^i(c_i) - \hat{V}_t^i(c_i - 1) + \sum_{k \in A_j, k \neq i} \pi_k~, \quad &\text{if } i \in A_j~, \\
V_t(c) - V_t(c - A_j) & \approx \sum_{k \in A_j} \pi_k~, \quad &\text{if } i \notin A_j~, \\
\end{align}

Mit $\Delta \hat{V}_t^i(c_i) := \hat{V}_t^i(c_i) - \hat{V}_t^i(c_i - 1)$ kann dies geschrieben werden als
$$ V_t(c) - V_t(c - A_j) \approx \left(\Delta \hat{V}_t^i(c_i) - \pi_i\right)\mathbb{1}_{i \in A_j} + \sum_{k \in A_j} \pi_k$$