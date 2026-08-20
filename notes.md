
"""
Gather corpus
Clean & preprocess data
    Remove HTML, navigation, boilerplate etc.
     
Decide chunking strategy - Contextual chunking
    Get an eval set (30-50 Q&A)
    Chunk the corpus - Start out with recursive splitter (500 tokens and 15% overlap) 
    Measure recall with eval set
    
    Maybe simpler? 
    
Decide model
    multilingual is a requirement 

Create vector database 
    psql & pgvector

Insert chunks and embeddings into database
    Chunk text
    Embedding 
    Document ID 
    Metadata 
    
Build retrieval 
    Convert user's query into embedding 
    Perform vector similarity search in pgvector (cosine)
    Return relevant chunks 
    
Build return model
    Quen:2.5:14b
    
Testing

Profit ???

Pipeline: 
Documents -> chunks -> embedding model -> vectors -< pgvector 

Query: 
Question -> embedding model - pgvector -> relevant chunks -> Quen2.5 -> answer

"""
