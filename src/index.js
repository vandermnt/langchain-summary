const ChatOpenAI = require('@langchain/openai');
const Prompts = require('@langchain/core/prompts');
const LanchChainSummarization = require('langchain/chains');
const { TokenTextSplitter } = require('langchain/text_splitter');

const llm = new ChatOpenAI.OpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  temperature: 0.9,
});

async function func() {
  const splitter = new TokenTextSplitter({
    chunkSize: 10000,
    chunkOverlap: 250,
  });

  const docsSummary = await splitter.createDocuments([
    ` Cliente: Olá! Estou em busca de alguns produtos para aprimorar meu setup. Poderia me passar os preços de alguns itens que vi em seu site?

  Vendedor: Olá! Claro, ficarei feliz em ajudar. Quais são os produtos específicos que você está interessado?
  
  Cliente: Estou de olho em um novo laptop, um teclado mecânico e também em um monitor ultrawide. Poderia me passar os preços desses itens?
  
  Vendedor: Certamente! Para o laptop modelo LMN, o preço atual é R$ 4.499,99. O teclado mecânico, modelo TKX, está por R$ 299,99, e o monitor ultrawide, modelo UWZ, está custando R$ 1.899,99.
  
  Cliente: Obrigado pelas informações. Vocês oferecem algum desconto para a compra de múltiplos itens ou algum pacote especial?
  
  Vendedor: Sim, temos promoções para compras combinadas! Se levar o laptop, teclado mecânico e o monitor juntos, você receberá um desconto adicional de 10% sobre o valor total da compra.
  
  Cliente: Isso é interessante! Além disso, há alguma garantia estendida disponível para esses produtos?
  
  Vendedor: Sim, oferecemos garantia estendida para todos os produtos. Para esses itens em específico, a garantia padrão é de 1 ano, mas você pode estendê-la para até 3 anos por uma taxa adicional. Isso proporcionará maior tranquilidade em relação à durabilidade e manutenção.
  
  Cliente: Excelente! Vou aproveitar a oferta e levar o laptop, teclado mecânico e o monitor ultrawide com o desconto de 10%. Também gostaria de adicionar a garantia estendida de 3 anos. Como procedo?
  
  Vendedor: Ótima escolha! Você pode adicionar os produtos ao carrinho no site e, durante o processo de checkout, o desconto será aplicado automaticamente. Para a garantia estendida, também será possível incluir essa opção durante o processo de compra. Se encontrar qualquer dificuldade, estou à disposição para ajudar.
  
  Cliente: Perfeito! Farei isso agora. Agradeço pela sua assistência!
  
  Vendedor: Eu que agradeço pela sua escolha! Se precisar de mais alguma ajuda ou tiver outras perguntas, não hesite em entrar em contato. Tenha uma ótima experiência de compra!`,
  ]);

  const text = Prompts.PromptTemplate
    .fromTemplate(`Resuma os dados dessa conversa do cliente com o vendedor:
    {text}
   `);

  const summarizeChain = LanchChainSummarization.loadSummarizationChain(llm, {
    type: 'refine',
    questionPrompt: text,
    refinePrompt: text,
    verbose: false,
  });

  const summary = await summarizeChain.run(docsSummary, {
    callbacks: [
      {
        handleLLMEnd: (output) => {
          const { completionTokens, promptTokens, totalTokens } =
            output.llmOutput?.tokenUsage;
          console.log(completionTokens ?? 0);
          console.log(promptTokens ?? 0);
          console.log(totalTokens ?? 0);
        },
      },
    ],
  });

  console.log(summary);
}

func();
