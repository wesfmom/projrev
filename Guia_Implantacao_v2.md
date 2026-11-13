# Guia de implantação — Agente de Revisão v2

Tempo estimado: **40 a 60 minutos**, sendo 30 deles preenchendo a base de conhecimento (que é o que faz o agente ficar bom).

Faça tudo logado na conta **allrevisao@grupoom.com.br**. Se você usa outra conta no mesmo navegador, abra uma janela anônima para não misturar.

---

## Etapa 0 — Backup (2 min) ⚠️ não pule

1. Abra o projeto atual em [script.google.com](https://script.google.com).
2. No arquivo de código, `Ctrl+A` → `Ctrl+C`.
3. Cole num Google Doc chamado **"Backup Agente Revisão v1"** e salve.
4. Ainda no editor, clique em **Acionadores** (ícone de relógio, barra da esquerda) e **anote quantos existem**. Você vai conferir isso no final.

> Se algo der errado em qualquer etapa, o rollback é: apagar o código novo, colar o backup, rodar `criarTrigger()` do v1. Menos de 2 minutos.

---

## Etapa 1 — Trocar o código (5 min)

O Apps Script compartilha o mesmo espaço de nomes entre todos os arquivos do projeto. **Se você deixar o código velho e adicionar o novo, as funções com o mesmo nome vão colidir e o comportamento fica imprevisível.** Então:

1. No editor, passe o mouse sobre o arquivo `Código.gs` na lista da esquerda → clique nos **três pontinhos** → **Excluir**.
2. Clique no **+** ao lado de "Arquivos" → **Script** → nomeie `AgenteRevisao`.
3. Apague o `function myFunction() {}` que vem por padrão.
4. Cole **todo** o conteúdo de `AgenteRevisao_v2.gs`.
5. `Ctrl+S`.

---

## Etapa 2 — Conferir as constantes do topo (3 min)

Logo no começo do arquivo, revise estas linhas:

```javascript
const CAIXA_EMAIL = 'allrevisao@grupoom.com.br';
const LABELS_ADICIONAIS = ['_Caixa de Entrada BB Zamp'];
const GEMINI_MODEL = 'gemini-3.7-flash';
```

- **`LABELS_ADICIONAIS`** — o nome tem que ser **exatamente** igual ao do Gmail, incluindo o underline inicial e os espaços. Se o marcador for aninhado (ex.: `Clientes/BB Zamp`), escreva com a barra.
- **`GEMINI_MODEL`** — se der erro 404 na Etapa 5, troque para `gemini-3.6-flash`.

Se quiser começar mais conservador (recomendado no primeiro dia), baixe temporariamente o lote:

```javascript
const MAX_THREADS_POR_EXECUCAO = 2;
```

---

## Etapa 3 — Serviços avançados (3 min)

Na barra da esquerda, ao lado de **Serviços**, clique no **+**:

| Serviço | Versão | Identificador |
|---|---|---|
| Drive API | **v3** | `Drive` |
| Gmail API | v1 | `Gmail` |

⚠️ **A versão do Drive precisa ser a v3.** Na v2 o método é `Drive.Files.remove` com outra assinatura e o código quebra.

Você já tinha esses dois ativados no v1 — só confirme que o Drive está mesmo em v3, porque projetos antigos às vezes ficam presos na v2.

---

## Etapa 4 — Chave da Gemini (1 min)

**Configurações do projeto** (ícone de engrenagem) → role até **Propriedades do script**.

Deve existir `GEMINI_API_KEY`. Se já existe do v1, não precisa mexer. Se não:

1. Gere em [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. **Adicionar propriedade do script** → nome `GEMINI_API_KEY`, valor = a chave.

As outras três propriedades (`PASTA_TEMP_ID`, `PLANILHA_BASE_ID`, `MSGS_PROCESSADAS`) são criadas sozinhas — não crie à mão.

---

## Etapa 5 — Rodar o `bootstrap()` (5 min)

1. No seletor de função (barra do topo, ao lado de ▶ Executar), escolha **`bootstrap`**.
2. Clique em **Executar**.
3. Vai aparecer **"Autorização necessária"** → **Revisar permissões** → escolha a conta `allrevisao@` → aparece a tela cinza **"O Google não verificou este app"** → clique em **Avançado** → **Acessar Agente de Revisão (não seguro)** → **Permitir**.

> Essa tela de "app não verificado" é normal para script próprio. Você está autorizando o seu próprio código.

4. As permissões pedidas incluem **Planilhas** agora (o v1 não usava) — por isso a tela de autorização reaparece mesmo que você já tenha autorizado antes.

**O que conferir no log** (`Ctrl+Enter` ou menu *Execuções*): deve terminar com

```
Bootstrap concluído. Planilha da base de conhecimento: https://docs.google.com/spreadsheets/d/XXXX/edit
```

**Abra esse link e favorite.** É a planilha que você vai usar dali em diante.

5. Volte em **Acionadores**. Devem existir **exatamente 2**: `verificarNovosEmails` (a cada 5 min) e `faxinaDiaria` (diário, ~3h). Se sobrou algum do v1, **apague manualmente**.

---

## Etapa 6 — `diagnostico()` (2 min)

Selecione **`diagnostico`** → **Executar**. O log deve sair assim:

```
GEMINI_API_KEY: ok
Drive API: ok
Gmail API: ok
Planilha base: https://docs.google.com/spreadsheets/d/...
Tamanho do system prompt: 2400 caracteres
Gemini: ok — 3 erro(s) no texto de teste
```

| Se aparecer | Faça |
|---|---|
| `Drive API: FALHOU` | Volte à Etapa 3 e confirme v3. Se persistir, ative a **Drive API** no [Cloud Console](https://console.cloud.google.com/apis/library/drive.googleapis.com) do projeto vinculado. |
| `Gmail API: FALHOU` | Idem, com a **Gmail API**. |
| `Gemini: FALHOU ... 404` | Nome do modelo. Troque para `gemini-3.6-flash`. |
| `Gemini: FALHOU ... 400 API key not valid` | Chave errada ou com restrição de IP/referrer. Gere uma nova sem restrições. |
| `Gemini: ok — 0 erro(s)` | Estranho: a frase de teste tem 3 erros. Revise se você colou o arquivo inteiro. |

**Não siga adiante enquanto essas cinco linhas não estiverem todas ok.**

---

## Etapa 7 — Teste controlado (10 min)

Antes de deixar o agente solto na caixa real:

1. Crie um Google Doc de teste com erros que você conhece — algo como:
   > *"Nos vamos analizar a proposta amanha. A equipe estão preparados para a reunião as 14h. O relatorio da Zamp foi enviado a diretoria."*
2. Envie um e-mail para `allrevisao@grupoom.com.br` com esse doc anexado (ou o link).
3. Espere até 5 minutos e confira a resposta na mesma conversa.

**O que deve acontecer:**

- ✅ Resposta na mesma thread, com caixa cinza de resumo no topo
- ✅ Cada erro num bloco com ID (`A1-E1`), tarja azul (confiança alta) ou amarela (confirmar)
- ✅ Nenhum `###` ou `**` aparecendo cru no corpo
- ✅ A conversa ganhou o marcador `IA-Processado`
- ✅ **"Zamp" NÃO foi apontado como erro** (já está no glossário de exemplo)

**Teste o follow-up** (era o bug mais grave do v1): responda a mesma thread anexando **outro** documento. Em até 5 min deve chegar uma segunda revisão. No v1 isso era silenciosamente ignorado.

**Teste o aprendizado:** responda a thread com **apenas** esta linha, no topo da mensagem:

```
#naocorrigir: Grupo OM | marca | razão social, nunca corrigir
```

Deve chegar um "Anotado." e o marcador `IA-Aprendizado`. Confira que a linha apareceu na aba `Glossario` da planilha.

> ⚠️ Um e-mail é **ou** comando **ou** revisão, nunca os dois. Se mandar comando junto com anexo, o anexo é ignorado. Mande em respostas separadas.
>
> ⚠️ Escreva o comando **no topo** da resposta, acima do texto citado.

---

## Etapa 8 — Preencher a base de conhecimento (30 min) — a parte que mais importa

Abra a planilha e preencha. Isto é o que separa um revisor genérico de um revisor do Grupo OM.

### Aba `Glossario` — prioridade máxima

Tudo que o agente **nunca** pode apontar como erro. Ataque isto primeiro: é a origem de quase todo falso positivo.

| termo | grafia_correta | tipo | observacao | ativo |
|---|---|---|---|---|
| Zamp | Zamp | marca | nunca acentuar | sim |
| Grupo OM | Grupo OM | marca | razão social | sim |
| BB | BB | sigla | Banco do Brasil | sim |
| Burger King | Burger King | marca | sem tradução | sim |

Puxe da cabeça: nomes de clientes, produtos, campanhas, praças, siglas internas, nomes de executivos que aparecem em assinatura. **Meta: 30 a 50 linhas.**

### Aba `RegrasDaCasa`

Preferências editoriais — coisas que não são erro de português, mas são erradas *para vocês*.

| regra | exemplo_errado | exemplo_certo | origem | ativo |
|---|---|---|---|---|
| Escrever "e-mail" com hífen | email | e-mail | manual do redator | sim |
| Não usar gerundismo | vamos estar enviando | enviaremos | manual do redator | sim |
| Datas por extenso em títulos | 12/05 | 12 de maio | padrão da agência | sim |

Se existir um manual de redação do Grupo OM, ele é a matéria-prima direta desta aba.

### Aba `Exemplos`

Deixe crescer sozinha pelos comandos de e-mail. Só preencha à mão se tiver casos polêmicos que você já sabe que o agente erra.

### Abas `Feedback` e `Log`

**Não edite.** São preenchidas pelo agente.

> Depois de editar a planilha à mão, rode `limparCachePrompt()` no editor para o efeito ser imediato. Sem isso, demora até 10 minutos.

---

## Etapa 9 — Ligar para valer (1 min)

1. Se você tinha baixado `MAX_THREADS_POR_EXECUCAO` para 2, volte para 5 e salve.
2. Rode `limparCachePrompt()`.
3. Pronto — o agente está operando.

---

## Rotina depois de no ar

### Diária (semana 1) — 2 minutos
Menu **Execuções** no editor. Olhe se há linhas vermelhas. Um `429` ocasional é normal (o retry resolve); erro repetido não é.

### Toda vez que uma revisão vier errada — 20 segundos
Responda a thread:

```
#falsopositivo: "à medida que" | locução correta, não leva crase
#passoubatido: "concerteza" -> "com certeza" | ortografia
```

**Esta é a rotina que faz o agente melhorar.** Se ninguém responder com feedback, ele fica parado no mesmo nível para sempre. Vale combinar com o time: *toda revisão que vier errada, um `#falsopositivo` de volta.*

### Mensal — 15 minutos
Rode `relatorioDeRecorrencias()` e leia o log. Os padrões que aparecem 3+ vezes viram linhas novas em `RegrasDaCasa`. É assim que você fecha o ciclo: erro recorrente do time → regra da casa → o agente passa a cobrar sozinho.

---

## Custo estimado

Com Gemini 3.7 Flash a US$ 0,75 / 1M tokens de entrada:

| Volume | Custo/mês aproximado |
|---|---|
| 20 documentos de ~20 páginas | menos de US$ 1 |
| 200 documentos | ~US$ 5 |
| 1.000 documentos | ~US$ 25 |

Ordem de grandeza — depende do tamanho real dos arquivos. O gargalo prático não é custo, é a cota de execução do Apps Script (90 min/dia em conta gratuita, 6 h/dia em Workspace).

---

## Problemas comuns depois de no ar

| Sintoma | Causa provável | Solução |
|---|---|---|
| Agente parou de responder | Trigger desativado após erros repetidos | Acionadores → confirme os 2; rode `criarTriggers()` |
| Responde duas vezes | Trigger duplicado do v1 sobrou | Apague os extras em Acionadores |
| "Não foi possível ler" em link do Drive | Arquivo sem permissão para `allrevisao@` | Compartilhe o arquivo/pasta com a conta |
| PDF escaneado só revisa o começo | OCR do Drive lê ~10 primeiras páginas | Envie o Google Doc convertido, não o PDF |
| Muito falso positivo | Glossário vazio | Etapa 8, aba `Glossario` |
| Comando `#regra` ignorado | Estava abaixo do texto citado, ou junto com anexo | Comando no topo, em e-mail separado |
| `Exceeded maximum execution time` | Pasta grande demais numa tacada | Baixe `MAX_ARQUIVOS_POR_EMAIL` para 6 |

---

## Checklist final

- [ ] Backup do v1 salvo num Doc
- [ ] Arquivo antigo **excluído** do projeto
- [ ] Drive API em **v3** e Gmail API ativos
- [ ] `bootstrap()` rodou e devolveu o link da planilha
- [ ] Exatamente **2** acionadores
- [ ] `diagnostico()` com as 5 linhas ok
- [ ] Teste com doc conhecido → resposta bem formatada
- [ ] Teste de follow-up na mesma thread → segunda revisão chegou
- [ ] Teste de `#naocorrigir` → gravou na planilha
- [ ] `Glossario` com 30+ linhas
- [ ] `RegrasDaCasa` preenchida
- [ ] Time avisado sobre os comandos `#falsopositivo` / `#passoubatido`
