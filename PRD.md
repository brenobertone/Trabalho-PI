# Product Requirements Document (PRD): Custom Branch-and-Bound para USApHMP-N

## 1. Visão Geral e Objetivo
Este documento detalha os requisitos de implementação de um solver exato customizado baseado em *Branch-and-Bound* para resolver o Problema de p-Hub Mediana Não-Capacitado de Alocação Única (USApHMP), utilizando a formulação baseada em fluxo multi-commodity (USApHMP-N). A implementação fará uso do solver `PuLP` para as relaxações lineares e incorporará uma meta-heurística de *Simulated Annealing* (SA) na raiz para garantir um forte *upper-bound* inicial.
Os dados de entrada estão disponíveis através do script dataset_parser.py

## 2. Preparação de Dados (USApHMP-N)
A formulação requer o cálculo de parâmetros de oferta e demanda consolidados para cada nó:
*   $O_i = \sum_{j \in N} W_{ij}$ (Oferta/Fluxo total saindo de $i$)
*   $D_i = \sum_{j \in N} W_{ji}$ (Demanda/Fluxo total entrando em $i$)

Os custos relativos são fatorados pelas constantes $\mathcal{X}$ (coleta), $\alpha$ (transferência) e $\delta$ (distribuição).

## 3. Heurística de Root-Node: Simulated Annealing
Para obter um limite superior ($V_{UB}$) de alta qualidade, será implementado o Algoritmo 1 do relatório.

**Parâmetros Básicos:**
*   **Comprimento da Cadeia de Markov:** $2np$ (onde $n$ é o total de nós e $p$ o de hubs).
*   **Fator de Resfriamento (Decaimento):** $0.97$.
*   **Probabilidades Iniciais de Operador:** $P(\text{Swap}) = 0.40$, $P(\text{Move}) = 0.60$.
*   **Limites:** Máximo de 5 cadeias consecutivas sem melhora ($c \le 5$); máximo de 4 reaquecimentos ($r \le 4$).

**Lógica de Execução:**
1. Gerar solução viável inicial $S$ e definir $S^* \gets S$.
2. Estimar $T_0$ (Temperatura Inicial) através do desvio padrão do custo em uma amostragem preliminar.
3. Enquanto $r \le 4$:
   4. Enquanto $c \le 5$:
      5. Para $i = 1$ até $2np$:
         * Selecionar operador de vizinhança conforme roleta de $P(\text{Swap})$ e $P(\text{Move})$.
         * Operadores:
           * **Swap:** Move um não-hub para outro cluster.
           * **Move:** Promove um não-hub a hub no mesmo cluster, rebaixa o hub atual. (Se cluster for singleton, acionar **Singleton Move**: escolhe nó de outro cluster para ser o novo hub isolado).
         * Calcular o Delta de custo incremental: $\Delta = \text{Custo}(S') - \text{Custo}(S)$.
         * Se $\Delta < 0$ ou $\text{Random}(0,1) < \exp(-\Delta / T)$:
           * $S \gets S'$ (Aceitação)
           * Se $\text{Custo}(S) < \text{Custo}(S^*)$, atualiza $S^* \gets S$ e $c \gets 0$.
      6. Atualizar Temperatura: $T \gets 0.97 \times T$.
      7. $c \gets c + 1$.
   8. Reaquecimento: $T \gets T_{\text{reaquecimento}}$, $P(\text{Swap}) \gets P(\text{Swap}) + 0.05$.
   9. $r \gets r + 1$, $c \gets 0$.
10. Retornar a melhor solução $S^*$. O custo desta solução se torna o $V_{UB}$ global.

## 4. Avaliação de Nó do Branch-and-Bound
A árvore de B&B executará as seguintes etapas em cada nó (Algoritmo 2):

**Entrada:** Nó atual da árvore, $V_{UB}$ global.
1. **Resolução LP:** Construir a relaxação linear (USApHMP-N) respeitando as fixações/restrições do nó atual e resolver usando `PuLP`.
2. **Poda por Inviabilidade ou Limite:**
   * Se a LP é inviável, ou se $V_{LP} \ge V_{UB}$, podar o nó (retornar vazio).
3. **Poda por Integralidade:**
   * Se a solução for inteira (todas variáveis $Z_{ik} \in \{0, 1\}$), atualizar $V_{UB} = \min(V_{UB}, V_{LP})$ e podar o nó.
4. **Fixação de Variáveis (Redução de Escopo):**
   * Extrair custos reduzidos $\bar{c}_{ik}$ da solução ótima da LP para variáveis não-básicas onde $Z_{ik} = 0$.
   * Se $V_{LP} + \bar{c}_{ik} > V_{UB}$, adicionar restrição para sub-árvore: $Z_{ik} = 0$ (consequentemente $Y_{ikl} = 0, \forall l$).
5. **Aplicação das Regras de Ramificação (Hierarquia Estrita):**
   * *Verificar se existem variáveis de hub ($Z_{kk}$) fracionárias.*
   * **Se SIM:**
     * **Condição BC1:** Avaliar pares $s, t$ de hubs fracionários. Se o par que maximiza a soma obedece $Z_{ss} + Z_{tt} > 1.0$:
       * Criar ramo 1: $Z_{ss} + Z_{tt} \ge 2$.
       * Criar ramo 2: $Z_{ss} + Z_{tt} \le 1$.
     * **Condição BC2:** Caso BC1 falhe, verificar se existe pelo menos um hub confirmado ($Z_{ss} = 1$). Achar $t$ fracionário que maximize $Z_{ss} + Z_{tt}$.
       * Criar ramo 1: $Z_{tt} = 1$.
       * Criar ramo 2: $Z_{tt} = 0$.
     * **Condição BC3:** Caso BC1 e BC2 falhem (variáveis fracionárias pequenas), escolher a variável de hub individual $Z_{kk}$ mais fracionária (próxima de 0.5):
       * Criar ramo 1: $Z_{kk} = 1$.
       * Criar ramo 2: $Z_{kk} = 0$.
   * **Se NÃO (Hubs estão todos inteiros, mas alocações $Z_{ik}$ são fracionárias):**
     * Aplicar **BC6** (A FAZER): Neste projeto, lançar um `NotImplementedError` ou aviso de que esta regra ficará em aberto.
6. Empilhar subproblemas gerados para avaliação.

## 5. Implementação Técnica
* **Linguagem:** Python 3.
* **Bibliotecas Principais:** `PuLP` (modelagem e LP), `math`, `random` (SA), `numpy` (matrizes e distâncias, caso aplicável).
* **Estrutura de Código:**
  * `sa_heuristic.py`: Implementação do Algoritmo 1.
  * `bb_engine.py`: Árvore de busca, gestão da pilha de nós e Algoritmo 2.
  * `lp_model.py`: Construção sob demanda da relaxação e extração de custos reduzidos via PuLP.
