# Relatório Técnico-Científico — Assistente Acadêmico IFPA (Agente RAG + ReAct)

**Componente:** Tópicos Avançados em Computação I — Bacharelado em Ciência da Computação, IFPA Campus Ananindeua
**Equipe:** _[nomes]_ · **Data:** 02/10/2026
**Repositório:** _[link do GitHub]_ · **Aplicação:** _[link do Railway]_

> Rascunho base para o relatório de 2 a 3 páginas. Ajustem o texto com as observações e prints da equipe e exportem em PDF.

## 1. Problema e solução

As normas acadêmicas do IFPA estão distribuídas em documentos extensos (o RDP tem 98 páginas e o PPC de Ciência da Computação, 108). Encontrar uma regra específica — frequência mínima, critérios de aprovação, estágio — é lento e sujeito a erro de interpretação. Desenvolvemos um assistente virtual que responde em linguagem natural, mas **somente com base nos documentos oficiais**, citando documento e página de cada afirmação.

## 2. Arquitetura do agente

O sistema tem dois pipelines desacoplados:

**Indexação (offline).** `PyPDFLoader` extrai o texto de 241 páginas de três documentos oficiais (Resolução CONSUP nº 945/2023, PPC v8 e Guia Acadêmico 2025). O texto de cada documento é normalizado (hifenização, espaços), concatenado (para não cortar artigos na virada de página) e fatiado pelo `RecursiveCharacterTextSplitter` em 480 chunks, cada um com a página (calculada pelo offset) e o artigo a que pertence; vetorizado com `text-embedding-3-small` (1536 dimensões) e persistido no ChromaDB local com métrica de cosseno. Cada chunk recebe um ID determinístico (hash de arquivo + página + posição), de modo que reindexar não duplica vetores.

**Consulta (online).** O agente é criado com `create_agent` do LangChain v1, que compila um grafo LangGraph implementando o ciclo ReAct: o modelo (`gpt-4o-mini`) produz um *Thought*, decide uma *Action* (chamada de ferramenta), recebe a *Observation* e repete até ter evidência suficiente para a resposta final. A memória de sessão é um `InMemorySaver` (checkpointer do LangGraph) indexado por `thread_id = sessao_id`, o que permite perguntas de acompanhamento ("e a partir de qual semestre?"). Um middleware (`wrap_model_call`) limita o histórico enviado ao modelo a 6 mil tokens, sem apagar a conversa salva.

O agente é exposto por uma API FastAPI (`POST /api/chat`) com interface web que exibe o log Thought/Action/Observation, e também por um modo terminal (`python -m app.cli`).

## 3. Desenho das ferramentas

| Ferramenta | Parâmetros | Retorno |
|---|---|---|
| `buscar_normas_ifpa` | `consulta`: pergunta reescrita em linguagem normativa; `pensamento`: justificativa da busca | Até 4 trechos `[n] documento \| página \| relevância` + texto, ou `SEM_EVIDENCIA` |
| `listar_documentos_oficiais` | — | Lista de documentos indexados |

A docstring funciona como **contrato** com o modelo: descreve quando usar a ferramenta (lista de assuntos acadêmicos), como formular a consulta (termos normativos, uma busca por subpergunta) e como interpretar o retorno (citar documento e página; não inventar diante de `SEM_EVIDENCIA`). O parâmetro `pensamento` torna o *Thought* explícito e auditável no log, já que modelos com tool calling nativo frequentemente omitem o raciocínio textual.

## 4. Estratégia de fatiamento (FinOps)

Adotamos `chunk_size = 1200` caracteres (~300 tokens) e `chunk_overlap = 200` (~17 %), com separadores hierárquicos da redação normativa (`TÍTULO`, `CAPÍTULO`, `Seção`, `Art.`, `§`) antes dos separadores genéricos. Justificativas:

- **Contexto:** a unidade de sentido de uma norma é o artigo com seus parágrafos, que tipicamente ocupa 600–1200 caracteres. 164 chunks começam exatamente em "Art." e 376 são rotulados com seu artigo; trechos que começam no meio de um artigo recebem o prefixo "(continuação do Art. N)". Na primeira versão (fatiamento por página) o agente atribuiu números de artigo errados a parágrafos soltos; o rótulo eliminou o problema.
- **Custo:** com `top_k = 4` (+3 do BM25), cada busca injeta ~1.200–2.000 tokens no prompt (≈ US$ 0,0003 no gpt-4o-mini). A indexação completa custa ≈ US$ 0,002 e, medido, uma pergunta completa custa de US$ 0,0006 a 0,0018 — valor exibido ao usuário após cada resposta.
- **Precisão do embedding:** chunks grandes misturam vários artigos e "diluem" o vetor; chunks pequenos separam a regra de suas exceções.
- **Overlap moderado:** garante que um parágrafo cortado na fronteira apareça íntegro em algum chunk, sem inflar o índice como um overlap de 50 % faria.

## 5. Mitigação de alucinações

1. **Ancoragem obrigatória:** o system prompt define os trechos recuperados como fonte única e exige busca antes de qualquer resposta normativa.
2. **Busca híbrida e limiar de relevância:** o BM25 complementa a busca vetorial com termos exatos (no teste, foi o que trouxe o Art. 88, "vedado o abono de faltas"); trechos com distância de cosseno acima de 0,75 são descartados; sem evidência, a ferramenta retorna `SEM_EVIDENCIA` e o agente declara que não encontrou a informação, orientando a procurar a Secretaria Acadêmica.
3. **Citação verificável:** a resposta final é um JSON com `citacoes` (documento, página, trecho). Após o json-repair e a validação Pydantic, cada citação é conferida contra as observações reais da sessão e todo número de artigo mencionado no texto precisa constar dos trechos recuperados; o que não confere é removido ou sinalizado e a confiança é rebaixada.
4. **Temperatura 0** e limite de 12 passos no grafo.
5. **Escopo e injeção:** filtro de entrada para padrões de prompt injection, recusa de temas fora do escopo e instrução de tratar o conteúdo recuperado como dado, não como ordem.

## 6. Tolerância a falhas e segurança

Exceções da OpenAI (autenticação, cota, timeout, conexão) e de recursão do grafo são capturadas e convertidas em respostas amigáveis com `status = "erro"`. A chave da API é lida apenas do ambiente (`.env` local, ignorado pelo Git, ou Variables do Railway). A aplicação roda em contêiner Docker com usuário sem privilégios, rate limit por IP e healthcheck.

## 7. Resultados e próximos passos

_[Inserir prints da demonstração: pergunta normativa com citação, pergunta de acompanhamento, pergunta sem evidência, tentativa de injeção bloqueada.]_

Próximos passos: incluir o Calendário Acadêmico 2026 e a Resolução nº 944/2023 (RDP da graduação); reranking com cross-encoder; memória persistente em banco (SqliteSaver/PostgresSaver); avaliação automática com um conjunto de perguntas-gabarito.
