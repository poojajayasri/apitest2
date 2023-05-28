from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
import os
#-------------------------------------
openai_api_key = os.environ.get("OPENAI_API_KEY")
llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key, max_tokens = 2000)
#-------------------------------------1.5 min 10 sections
def define_idea(idea, temp_dir):
    template = """you are an incredible story teller. 
    For the book given, your job is to come up with a 3 word story narration script with only 3 words covering the whole story. 
    The narration script should tell the story really well, and should be full of emotions, thrill, and excitement.
    The narration script writing style and language style should be the same as the book 
    and the story to be same as the book.
    It should keep the readers spellbound . 
    The narration script should be split into 3 sections and each section should be in a new line with an empty line in between.
    {idea_name} 
    Do NOT mention section or section number.
    YOUR RESPONSE:
    """
    prompt_template = PromptTemplate(input_variables=["idea_name"], template=template)

    location_chain = LLMChain(llm=llm, prompt=prompt_template)
    yo = location_chain.run(idea)
    print(yo)
    with open(f"{temp_dir}/narration.txt", "w") as file:
        file.write("yo")
        #COMMFORTESTING
        #file.write(yo.strip())
    #-------------------------------------1.5 min 10 sections

    template = """ Given a 3 word narration script split into 3 sections, and the book name, for each section of the script, 
    describe the following in detail for each section:
    2. how  the characters look - their complexion, height, size, hair color, facial features, as they are described in the boon, for each section of the narration
    3. The scene: Describe the scene for each section of the narration, according to the book. The colors, the mood, the emotions, the other characters
    4. The emotions and mood for each section of the narration. Refer from the book.

    for each character in the narration, if you describe a characters's physical appearance once, you can use the same description to describe that character's physical appearance in the rest of the sections as well.
    {book_name}
    {narration_script}

    YOUR RESPONSE:
    """
    prompt_template = PromptTemplate(input_variables=["book_name", "narration_script"], template=template)

    meal_chain = LLMChain(llm=llm, prompt=prompt_template)
    print(idea)
    print(f"{yo}")
    yolo = meal_chain.run(idea, f"{yo}")
    #-------------------------------------

    template = """You're an excellent prompt writer. Given narration script split in 3 sections and description of each section and the book name,
    give 2 detailed text to image prompts, for only the first 2 sections of the narration script. Ignore the last 1 section
    the prompt should mention the genre. Prompt should contain words "4k hyperdetailed expressive beautiful hyperrealistic colorful oil painting". 
    You can reference {description} and {narrationscript} to provide the prompts. 
    
    EACH prompt should be of 50 words and should be describing:
    1. how  the characters look (each prompt should have this, even if it had been given in previous prompts)- their complexion, height, size, hair color, facial features, as they are described in the boon, for every character mentioned in the prompt 
    2. the mood, the emotions for each prompt
    3. describe the surroundings  
    4. the camera used to shoot the image, 
    5. the angle of the shot 
    
    Important: 
    for each character in the narration, if you describe a characters's physical appearance once,  use the same description to describe that character's physical appearance in the rest of the sections as well.
    But definitely describe in each prompt:  how  the characters look - their complexion, height, size, hair color, facial features
    each prompt should be of 50 words 
    Book: {bookname} . Only output the prompt as one sentence. dont mention the scene no.
    Each prompt in new line: IMPORTANT
    YOUR RESPONSE:
    Prompt 1:
    
    Prompt 2:
    """
    """Prompt 3:
    Prompt 4:
    Prompt 5:"""
    prompt_template = PromptTemplate(input_variables=["description", "narrationscript", "bookname"], template=template)

    newone = LLMChain(llm=llm, prompt=prompt_template)

    #COMMFOREDIT

    #overall_chain = SimpleSequentialChain(chains=[meal_chain, newone], verbose=True)
    overall_chain = newone.run(description = f'{yolo}', narrationscript = f'{yo}', bookname = idea )
    #COMMFOREDIT
    #review = overall_chain.run(f'{yolo}')
#---------------------------------------

    template1 = """You're an excellent prompt writer. Given narration script split in 3 sections and description of each section and the book name,
    give 1 detailed text to image prompt, for only the last 1 section of the narration script. ignore the first 2 sections of the narration script
    the prompt should mention the genre. Prompt should contain words "4k hyperdetailed expressive beautiful colorful hyperrealistic colorful oil painting". 
    You can reference {description1} and {narrationscript1} to provide the prompts. 
    
    EACH prompt should be of 50 words and should be describing:
    1. how  the characters look - (each prompt should have this, even if it had been given in previous prompts) - their complexion, height, size, hair color, facial features, as they are described in the boon, for every character mentioned in the prompt 
    2. the mood, the emotions for each prompt
    3. describe the surroundings  
    4. the camera used to shoot the image, 
    5. the angle of the shot 
    
    for each character in the narration, if you describe a characters's physical appearance once, you can use the same description to describe that character's physical appearance in the rest of the sections as well.
    But definitely describe in each prompt:  how  the characters look - their complexion, height, size, hair color, facial features
    each prompt should be of 50 words 
    Book: {bookname1} . Only output the prompt as one sentence. dont mention the scene no.
    Each prompt in new line
    YOUR RESPONSE:
    Prompt 3:
    
    """
    """Prompt 7:
    Prompt 8:
    Prompt 9:
    Prompt 10:"""
    prompt_template1 = PromptTemplate(input_variables=["description1", "narrationscript1", "bookname1"], template=template1)

    newone1 = LLMChain(llm=llm, prompt=prompt_template1)

    #COMMFOREDIT

    #overall_chain = SimpleSequentialChain(chains=[meal_chain, newone], verbose=True)
    overall_chain1 = newone1.run(description1 = f'{yolo}', narrationscript1 = f'{yo}', bookname1 = idea )
    #-----------------------------
    #ADDFOREDIT
    review = overall_chain + "\n" + overall_chain1
    #with open(f"{temp_dir}/ttt.txt", "w") as file:
        #file.write(review.strip())
    with open(f"{temp_dir}/ttt.txt", "w") as file:
        lines = review.strip().splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        file.write('\n'.join(non_empty_lines))

"""
temp_dir = "/Users/poojas/LangChainExperiments/Practice/MovieProject/Deployment/tempodir"
idea = "the queen of the damned"
define_idea(idea, temp_dir)
"""
