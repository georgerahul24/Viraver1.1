import requests
from bs4 import BeautifulSoup
from langchain_community.llms import Ollama
from langchain.chains import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter

# Fetch the webpage content
url = 'https://en.wikipedia.org/wiki/International_Institute_of_Information_Technology,_Hyderabad'
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')

# Extract main content
main_content = soup.find('div', class_='mw-parser-output')
text = main_content.get_text(separator="\n")

# Initialize the language model
llm = Ollama(model="llama3")

# Split the text into chunks
splitter = CharacterTextSplitter(separator="\n\n", chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)

# Initialize the summarization chain
chain = load_summarize_chain(llm, chain_type="map_reduce")

# Summarize each chunk and combine the results
summaries = [chain.invoke(chunk) for chunk in chunks]
final_summary = "\n".join(summaries)

# Output the final summary
print(final_summary)
