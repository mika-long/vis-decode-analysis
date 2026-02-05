library(jsonlite)
library(purrr)
library(jsonvalidate)
library(httr)
library(here)

# --- initialize json validator 
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

BaseComponent <- function(type, path, instruction, instructionLocation, nextButtonLocation, parameters, response_list = NULL) {
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

# --- construct basecomponent 
moritz_component <- BaseComponent(
  type="react-component",
  path="vis-decode-retrieve-value/assets/Moritz.tsx",
  instruction= "Drag the slider to where the values are on average; imagine that this is some stock prices",
  instructionLocation = "aboveStimulus",
  nextButtonLocation = "belowStimulus",
  parameters = list(
    taskid = "Moritz",
    taskType = "moritz"
  ),
  response_list = list(moritz_response)
)
# test 
convert_to_json(moritz_component)

# creating a simple component 
task1 = Component(baseComponent = "Moritz", parameters = list(params = list(index = 4)))
convert_to_json(task1)

# creating a sequence of components 
components <- point_ids %>% imap(~ Component(
  baseComponent = "Moritz", 
  parameters = list(params = list(index = .x))
)) %>% 
  set_names(paste0("task", seq_along(point_ids)))

# build the sequence 
task_names = paste0("task", 1:48)
sequence <- list(
  order = "fixed", 
  components = list(
    "introduction", 
    "$virtual-chinrest.se.full",
    list(
      order = "random",
      components = c(
        task_names
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
json_output <- toJSON(card_component, auto_unbox = TRUE, pretty = TRUE)


final_output = list(
  `$schema` = "https://raw.githubusercontent.com/revisit-studies/study/v2.3.1/src/parser/StudyConfigSchema.json",
  studyMetadata = metadata, 
  uiConfig = ui_config,
  importedLibraries = list("virtual-chinrest"), 
  baseComponents = list(
    Moritz = moritz_component
  ), 
  components = components,
  sequence=sequence
)

config_json <- final_output %>% convert_to_json(.)
# validate 
validator(config_json, verbose = TRUE)

# save output 
config_json %>% write(here("R", "config.json"))

