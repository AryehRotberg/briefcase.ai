import re
from typing import Dict, List

import chromadb
from tqdm import tqdm


class ChromaBulletExtractor:
    def __init__(self, chroma_collection: chromadb.Collection, similarity_threshold: float) -> None:
        """
        Initialize the processor using ChromaDB for vector similarity search.
        
        Args:
            chroma_collection: ChromaDB collection instance
            similarity_threshold: Minimum similarity score to consider a match
        """
        self.similarity_threshold = similarity_threshold
        self.chroma_collection = chroma_collection
        self.batch_size = 100
        self.top_k_categories = 1
        self.section_patterns = [
            r'\n\s*#+\s+.*?\n',  # Markdown headers
            r'\n\s*\d+\.\s+',    # Numbered lists
            r'\n\s*[A-Z][^.]*:\s*\n',  # Title-like patterns
            r'\n\s*\*\*.*?\*\*\s*\n',  # Bold section headers
        ]

    
    def extract(self, document: str, show_progress: bool = True) -> Dict[str, List[str]]:
        """
        Process document using ChromaDB's native batch query for optimal performance.
        
        Args:
            document: Document text to process
            verbose: Whether to print detailed logs
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping categories to document chunks
        """
        chunks = self._split_document(document)
                
        result = {}
        
        iterator = range(0, len(chunks), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator,
                            desc="Processing batches",
                            total=len(chunks) // self.batch_size + 1)

        for batch_start in iterator:
            batch_end = min(batch_start + self.batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            try:
                batch_results = self.chroma_collection.query(
                    query_texts=batch_chunks,
                    n_results=self.top_k_categories
                )
                
                for idx, chunk in enumerate(batch_chunks):
                    if batch_results["documents"][idx]:
                        for k in range(len(batch_results["documents"][idx])):
                            category = batch_results["documents"][idx][k]
                            distance = batch_results["distances"][idx][k]
                            
                            similarity = 1 - (distance / 2)
                            
                            if similarity >= self.similarity_threshold:
                                if category not in result:
                                    result[category] = []
                                
                                result[category].append({
                                    'text': chunk,
                                    'similarity': similarity,
                                })
                
            except Exception as e:
                print(f"Error processing batch starting at {batch_start}: {e}")
                continue
        
        for category in result:
            result[category].sort(key=lambda x: x['similarity'], reverse=True)
        
        return result


    def _split_document(self, document: str) -> List[str]:
        """
        Split document into meaningful chunks that preserve context.
        Uses multiple strategies to identify natural breakpoints.
        """
        chunks = []
        current_text = document
        
        for pattern in self.section_patterns:
            matches = list(re.finditer(pattern, current_text))
            if matches:
                sections = []
                last_end = 0
                
                for match in matches:
                    if last_end < match.start():
                        sections.append(current_text[last_end:match.start()].strip())
                    sections.append(match.group().strip())
                    last_end = match.end()
                
                if last_end < len(current_text):
                    sections.append(current_text[last_end:].strip())
                
                current_text = '\n'.join([s for s in sections if s])
        
        paragraphs = re.split(r'\n\s*\n', current_text)
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            if len(paragraph) > 500:
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
                
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk + " " + sentence) > 300 and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(paragraph)
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 20]
