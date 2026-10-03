
# AI_LOG — uso de IA no PA2

**Ferramentas:** Claude, usado sem commitar código gerado sem revisão.

**Em uma frase:** usamos a IA como par para discutir conceitos, depurar e produzir código de apoio (gráficos, scripts de análise, documentação).

## Onde usamos e onde não usamos

| Tarefa | IA? | Nosso papel |
|---|---|---|
| Escolhas de projeto (Trilha A, Eixo 2, Parte 5, detector SDP, MOT17-09 como held-out, eixo de densidade) | Não | Decidimos com base em experimentos (`compare_detectors.py`, correlação densidade × razão de contagem) |
| Modelo (LSTM residual), perda, associação, nascimento e morte de tracks | Discussão de alternativas | Escrevemos e testamos o código |
| IDF1, ID switches, fragmentações | Explicação do Húngaro e das definições | Escrevemos a métrica e os casos de teste à mão |
| Gerador sintético com oclusão, simulador de detector | Sugestão de abordagem | Escrevemos, calibramos e verificamos com figuras |
| Figuras (matplotlib), estilo, scripts de análise | Sim, bastante | Revisamos e ajustamos os resultados |
| Interpretação dos resultados e diagnósticos da Parte 4 | Não | É a base da apresentação |
| README, `metrics.py` consolidado, `inferencia.ipynb` | Sim (ver episódio 6) | Revisamos e executamos |

## Exemplos de episódios de uso:

**1. Oclusão "de verdade" no gerador sintético (Parte 0).** O enunciado exige que uma elipse passe *atrás* de outra e desapareça. Discutimos com a IA como garantir isso: ordem de profundidade fixa, oclusor maior que o ocluído nos dois eixos e folga calibrada para a oclusão durar exatamente `occlusion_duration` quadros (`synthetic_video.py`). Escrevemos `verify_occlusion.py` para provar o efeito, separando "fora da tela" de "ocluído" ao medir a visibilidade. Sem essa separação, a visibilidade mínima mistura os dois casos.

**2. Curva de oclusão contaminada por outras tracks (Parte 0).** Na curva "IDF1 × duração da oclusão", o IDF1 de um objeto saía misturado com as tracks dos outros objetos da cena. Conversamos sobre o que o IDF1 penaliza (quadros previstos sem par viram IDFP). Chegamos à decisão de avaliar só as tracks previstas que chegam a sobrepor o objeto de interesse (`filter_relevant_tracks` em `stress_synthetic_baseline.py`). A decisão e o critério são nossos; a IA ajudou a enxergar a causa.

**3. IDF1, switches e fragmentações (Parte 0 e 1).** Pedimos à IA que explicasse a diferença entre atribuição global (IDF1) e atribuição por quadro (switches) e o que separa um switch de uma fragmentação. Antes de rodar qualquer coisa, calculamos no papel o resultado esperado de cada caso de teste: (a) IDF1 = 1, (b) troca em *k* gera 2 switches e IDF1 = 0,5, (c) track partida com lacuna dá IDF1 ≈ 0,44, 0 switches e 1 fragmentação. Definimos que só conta switch entre quadros consecutivos e documentamos a diferença para o CLEAR-MOT. Na entrega final, a IA reescreveu o `metrics.py` indexado por quadro (mais rápido). Conferimos que ele dá exatamente os mesmos números que a versão original em centenas de cenários aleatórios, com empates e lacunas.

**4. Descasamento treino × inferência no tracker (Parte 4).** Ao investigar a memória do LSTM, percebemos que, no tracker, a última caixa era consumida duas vezes (uma ao prever, outra ao recalcular o estado com a observação), enquanto no treino cada caixa entra uma única vez. Raciocinamos com a IA sobre o protocolo correto: *cada x_t entra uma vez; a entrada do passo seguinte é a observação casada ou, sob oclusão, a própria previsão*. Corrigimos o tracker e mantivemos a flag `legacy_state_update` só para reproduzir os números antigos.

**5. Definição de "falha" na galeria (Parte 4).** A primeira regra de "track roubada" (qualquer IoU ≥ 0,3 com outro GT) marcava quase toda oclusão como roubo, porque o oclusor sobrepõe o alvo por definição. Corrigimos para "a caixa observada explica *melhor* outro GT do que o próprio alvo". Excluímos da estatística os eventos sem track prévia (não testam memória) e resumimos a sobrevivência do estado com Kaplan-Meier, tratando recuperados como censurados. Ao medir o gradiente em GPU, o `backward` de RNN em modo `eval` falhou no cuDNN. Colamos o erro na IA e aceitamos a solução (desligar o cuDNN só dentro da medida) porque muda o kernel usado, não a matemática do gradiente. Também trocamos as cores das figuras por um mapa local por figura, porque IDs diferentes (por exemplo 1 e 81) caíam na mesma cor por módulo 20.

**6. Entrega final.** Pedimos à IA o rascunho do README, da consolidação do `metrics.py` e do `inferencia.ipynb`. Revisamos cada comando do README contra os scripts, rodamos o notebook em uma sequência de treino e em uma de teste, e conferimos o vídeo e a contagem.

## Como verificamos o que a IA produziu

- **Testes com resposta conhecida:** `scripts/tests/` (métricas em casos feitos à mão, piso fácil do baseline, degradador e mAP).
- **Baseline trivial:** `sanity_check_motion_lstm.py` compara o LSTM com "copiar a última posição". Um modelo que não bate esse baseline não aprendeu movimento.
- **Split por sequência:** a MOT17-09 nunca entra no treino de `motion_lstm.pt`.
- **Código gerado:** nada foi usado sem ler, rodar e entender. 
