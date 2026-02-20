library(jsonlite)
library(purrr)
library(jsonvalidate)
library(httr)
library(here)
library(tidyverse)

# --- initialize json validator
# --- only works when one have access to the internet 
schema_url <- "https://raw.githubusercontent.com/revisit-studies/study/v2.3.1/src/parser/StudyConfigSchema.json"
schema <- content(GET(schema_url), as = "text")
validator <- json_validator(schema, engine = "ajv")

# --- Get the list of numbers to pull from
exp_json_data <- jsonlite::fromJSON(here("data", "moritz", "stimuli", "exp2-stimuli.json" )) %>% as_tibble(.)

stimuli_df <- exp_json_data %>% select(index, seed, noise, flip, type, data) %>% 
    mutate(noise = as.character(noise))

point_ids <- stimuli_df %>% filter(type == "point") %>% 
  pull(index)

pointArc_ids <- stimuli_df %>% filter(type == "point_arc") %>% 
  pull(index)

#--- Define UI and metada 

metadata <- list(
  title = "Visual Decoding Operators (Retrieve Value)",
  version = "pilot",
  authors = list("Sheng Long"),
  date = "2025-12-10",
  description = "A study designed to elicit resposnes from participants", 
  organizations = list("Northwestern University")
)

ui_config = list(
  contactEmail = "shenglong@u.northwestern.edu", 
  helpTextPath = "vis-decode-retrieve-value/assets/help.md",
  logoPath = "revisitAssets/revisitLogoSquare.svg",
  withProgressBar = TRUE, 
  autoDownloadStudy = FALSE,
  withSidebar = FALSE,
  minHeightSize = 400,
  minWidthSize = 400, 
  urlParticipantIdParam = "PROLIFIC_PID",
  studyEndMsg = "**Thank you for completing the study.** \n\n You may click this link and return to Prolific: [https://app.prolific.com/submissions/complete?cc=C1O3ZUHV](https://app.prolific.com/submissions/complete?cc=C1O3ZUHV)"
)


#--- define functions 
StudyResponse <- function(id, type, prompt, required = FALSE, hidden = FALSE) {
  list(
    id = id,
    prompt = prompt,
    required = required,
    type = type,
    hidden = hidden
  )
}

# 'sidebar' | 'aboveStimulus' | 'belowStimulus' | 'stimulus'
BaseComponent <- function(
  type, 
  path, 
  instruction, 
  instructionLocation, 
  nextButtonLocation, 
  parameters, 
  response_list = NULL) {
  list(
    type = type,
    path = path,
    instruction = instruction, 
    instructionLocation = instructionLocation, 
    nextButtonLocation = nextButtonLocation,
    parameters = parameters,
    response = if (is.null(response_list)) list() else response_list
  )
}

Component <- function(baseComponent, parameters){
  list(
    baseComponent = baseComponent,
    parameters = parameters
  )
}

convert_to_json <- function(c){
  toJSON(c, auto_unbox = TRUE, pretty = TRUE)
}

# --- construct response 
moritz_response = StudyResponse(
  id="weightedAverage-value",
  prompt="The value of response",
  required = FALSE,
  type = "numerical",
  hidden = TRUE
)
# --- test 
moritz_response %>% convert_to_json(.)

# --- construct basecomponent 
moritz_component <- BaseComponent(
  type="react-component",
  path="vis-decode-retrieve-value/assets/Moritz.tsx",
  instruction= "***Experiment Instructions.*** Please read the following paragraphs carefully. You will be asked questions about the information in the paragraphs. \n***Scenario:*** Assume that you are a stock market investor. You are investing your own money in stocks, and you want to determine the average price of a stock over time in order to pick the best investment.\n ***Task:*** In this experiment, you will be shown graphs of stock prices over a one-year period like the one below. Your task is to determine the average stock price for that year. What is the average stock price? (Click and drag the line to indicate the average stock price)\n ***Response:*** To indicate the average stock price, use your mouse to drag the line on the chart. Move the line to where you think the average stock price is for that year. You can readjust the line by clicking and dragging. Once you are happy with your judgment of the average stock price, click the next button.",
  instructionLocation = "aboveStimulus",
  nextButtonLocation = "belowStimulus",
  parameters = list(
    taskid = "Moritz",
    taskType = "moritz"
  ),
  response_list = list(moritz_response)
)

# --- test 
moritz_component %>% convert_to_json()

# creating a simple component 
task1 = Component(baseComponent = "Moritz", parameters = list(params = list(index = 4)))
# --- test 
convert_to_json(task1)

# create the component for introduction 
# "introduction": {
#             "path": "vis-decode-retrieve-value/assets/introduction.md",
#             "response": [],
#             "type": "markdown"
#  },
introduction_component <- list(
  introduction = list(
    path = "vis-decode-retrieve-value/assets/introduction.md",
    response = list(),
    type = "markdown"
  )
)
# test 
introduction_component %>% convert_to_json(.)

# creating a sequence of components 
# for point 
components <- point_ids %>% imap(~ Component(
  baseComponent = "Moritz", 
  parameters = list(params = list(index = .x))
)) %>% 
  set_names(paste0("point_task_", seq_along(point_ids)))
# creating a sequence of components 
# for point arcs 
point_arc_components <- pointArc_ids %>% imap(~ Component(
  baseComponent = "Moritz", 
  parameters = list(params = list(index = .x))
)) %>% 
  set_names(paste0("pointArc_task_", seq_along(pointArc_ids)))

# build the sequence 
point_task_names = paste0("point_task_", 1:48)
pointArc_task_names = paste0("pointArc_task_", 1:48)
sequence <- list(
  order = "fixed", 
  components = list(
    "introduction", 
    "$virtual-chinrest.se.full",
    list(
      order = "latinSquare", # needs to be latin square to ensure sampling efficiency 
      numSamples = 1, # randomly select 1 
      # components = c(
      #   point_task_names
      # )
      components = list(
        list(
          order = "random", 
          components = c(point_task_names)
        ), 
        list(
          order = "random", 
          components = c(pointArc_task_names)
        )
      )
    )
  )
)
sequence %>% convert_to_json(.)

# 4. RENDER
# the final output has the following components: 
# 1. schema 
# 2. studyMetadata 
# 3. uiConfig, 
# 4. importedLibraries 
# 5. baseComponents 
# 6. components 
# 7. sequence 

final_output = list(
  `$schema` = "https://raw.githubusercontent.com/revisit-studies/study/v2.3.1/src/parser/StudyConfigSchema.json",
  studyMetadata = metadata, 
  uiConfig = ui_config,
  importedLibraries = list("virtual-chinrest"), 
  baseComponents = list(
    Moritz = moritz_component
  ), 
  # components = components,
  components = c(introduction_component, components, point_arc_components),
  sequence=sequence
)

config_json <- final_output %>% convert_to_json(.)
# validate 
validator(config_json, verbose = FALSE)

# save output 
config_json %>% write(here("R", "config-r.json"))

