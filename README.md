# 1N20

Can you tell me the current sales agent in my code uses a tool which gets 
the data from the rag corpus. Can you tell me if i want to the sales agent 
to use this json as a tool, and get only the relavant item data from it, how
 can should this be created? Dont code

 
what i want to do is lets say i have stored in mongo db, the agent can hit a tool to first get the knowledge about what all data folders are there in the collection, when it hit this 
it gets an idea about all the folders and a short descirption about what data they have in the folder, the agent then loads this in its memory, and now whenever it gets a user query then
 it checks which folder will have the most relevant data about the query it can be access more than one folders as well, in each folder we might have sub folders for different data about
 the item of which the folder is about, so that it can get the most necesary data only without getting extra uncessary information. Can you tell me how can we do it, just brain storm 
with me. Dont code about how we can do this 




Ok now what i want you to do is, you got an idea of what is getting embedded
  and what kind of data is in the metadata, now what i want you to do is.
  Make changes in the sales agent and search product tol both 1. As you know
  what is getting embedded modify the sales agent so that it reframes the
  asked question in such a way that it can easily get the most relevant chunk
  from the vector database.  2. Next update the serach product tool, so that
  it comprehesively uses the metadata as you got an idea. and formats it along
  with the retrieved chunk to provide the model maximum informatation. Make
  sure the data which is added along with the chunk from the metadata is
  already not present in the description part which is retrived. Or wha 