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
  title = "Visual Decoding Operators — Average Estimation in Line Graphs",
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
moritz_num_response = StudyResponse(
  id="numericResponse",
  prompt="Numeric response of slider to Mortiz question",
  required = FALSE,
  type = "numerical",
  hidden = TRUE
)
moritz_px_response = StudyResponse(
  id="pixelResponse",
  prompt="Numeric response of slider to Mortiz question",
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
  instruction= "***Experiment Instructions.*** Please read the following paragraphs carefully. You will be asked questions about the information in the paragraphs. \n\n***Scenario:*** Assume that you are a stock market investor. You are investing your own money in stocks, and you want to determine the average price of a stock over time in order to pick the best investment.\n\n***Task:*** In this experiment, you will be shown graphs of stock prices over a one-year period like the one below. Your task is to determine the average stock price for that year. **What is the average stock price?** \n\n***Response:*** To indicate the average stock price, use your mouse to click and drag the handle on the slider, which will show a dotted red line. Move the line to where you think the average stock price is for that year. You can readjust the line by clicking, dragging, or using the left and right arrow keys. Once you are happy with your judgment of the average stock price, click the next button.",
  instructionLocation = "aboveStimulus",
  nextButtonLocation = "belowStimulus",
  # commetning these out
  # parameters = list(
  #   taskid = "Moritz",
  #   taskType = "moritz"
  # ),
  response_list = list(moritz_num_response, moritz_px_response)
)

# --- test 
moritz_component %>% convert_to_json()

# creating a simple component 
task1 = Component(baseComponent = "Moritz", parameters = list(params = list(index = 4)))
# --- test 
convert_to_json(task1)

# create the component for introduction 

introduction_component <- list(
  introduction = list(
    path = "vis-decode-retrieve-value/assets/introduction.md",
    response = list(
      list(
        id = "prolificId",
        prompt = "Please enter your Prolific ID (without any spaces):",
        required = TRUE,
        location = "belowStimulus",
        type = "shortText", 
        placeholder = "Prolific ID",
        paramCapture = "PROLIFIC_PID"
      )
    ),
    type = "markdown"
  )
)
# test 
introduction_component %>% convert_to_json(.)

# create the document for consent 
consent_comp = list(
  consent = list(
    type = "markdown",
    path = "vis-decode-retrieve-value/assets/consent.md", 
    # nextButtonText = "I agree",
    response = list(
      list(
        id = "consentApproval",
        prompt = "Do you consent to the study and wish to continue?", 
        required = TRUE,
        location = "belowStimulus", 
        type = "radio",
        options = c("Decline", "Accept")
      )
    )
  )
)
# test 
consent_comp %>% convert_to_json(.)

# attention check component
attention_check = list(
  attention_check = list(
    type = "markdown",
    path = "vis-decode-retrieve-value/assets/attention_check.md",
    response = list(
      list(
        id = "attention_check",
        prompt = "During this study, you will be asked to look at graphs of _____ prices. \n\n What should _____ be?",
        required = TRUE,
        location = "belowStimulus",
        type = "shortText"
      )
    )
  )
)

calibration_intro = list(
  calibration_intro = list(
    type = "markdown",
    path = "vis-decode-retrieve-value/assets/calibration_intro.md", 
    response = list(),
    nextButtonEnableTime = 3000
  )
)

training_intro = list(
  training_intro = list(
    type = "markdown",
    path = "vis-decode-retrieve-value/assets/training_intro.md", 
    response = list(),
    nextButtonEnableTime = 3000
  )
)

testing_intro = list(
  testing_intro = list(
    type = "markdown",
    path = "vis-decode-retrieve-value/assets/testing_intro.md", 
    response = list(),
    nextButtonEnableTime = 3000
  )
)

# create post study survey component 
post_study_component = list(
  post_study = list(
  type = "questionnaire",
  response = list(
    list(
      id = "gender", 
      prompt = "## Thank you! \n\n Please answer the following demographics related questions: \n\n Gender",
      location = "belowStimulus",
      type = "radio",
      options = c("Male", "Female", "Prefer to self describe", "Prefer not to say")
    ), 
    list(
      id = "self-gender", 
      prompt = "If you have selected 'Prefer to self describe', what is your gender?",
      location = "belowStimulus",
      type = "shortText",
      required = FALSE
    ), 
    list(
      id = "age",
      prompt = "What is your age?",
      location = "belowStimulus",
      type = "numerical"
    ), 
    list(
      id = "feedback", 
      prompt = "Were any of the instructions unclear?",
      type = "radio",
      options = c("Yes", "No")
    ), 
    list(
      id = "feedback-text",
      prompt = "(Optional) If so, which instructions were unclear?", 
      location = "belowStimulus",
      type = "longText",
      required = FALSE
    ), 
    list(
      id = "strategy",
      prompt = "(Optional) Please share with us any **strategy** you used.",
      location = "belowStimulus",
      type = "longText",
      required = FALSE
    )
  )
)
)

# test 
post_study_component %>% convert_to_json(.)

# ---

# creating a sequence of components 
# for point 
components <- point_ids %>% imap(~ Component(
  baseComponent = "Moritz", 
  # parameters = list(params = list(index = .x, type = "point"))
  parameters = list(taskIndex = .x, taskType = "point")
)) %>% 
  set_names(paste0("point_task_", seq_along(point_ids)))
# creating a sequence of components 
# for point arcs 
point_arc_components <- pointArc_ids %>% imap(~ Component(
  baseComponent = "Moritz", 
  # parameters = list(params = list(index = .x, type = "pointArc"))
  parameters = list(taskIndex = .x, taskType = "pointArc")
)) %>% 
  set_names(paste0("pointArc_task_", seq_along(pointArc_ids)))

# build the sequence 
point_task_names = paste0("point_task_", 1:48)
pointArc_task_names = paste0("pointArc_task_", 1:48)
sequence <- list(
  order = "fixed", 
  components = list(
    "introduction", 
    list(
      components = list("consent"),
      skip = list(list(
        name = "consent",
        check = "response",
        responseId = "consentApproval",
        value = "Decline",
        to = "end",
        comparison = "equal"
      )), 
      order = "fixed"
    ),
    "calibration_intro",
    "$virtual-chinrest.se.full",
    "training_intro", # training intro 
    # missing training section here
    "testing_intro",
    list(
      order = "latinSquare", # needs to be latin square to ensure sampling efficiency 
      numSamples = 1, # randomly select 1 
      components = list(
        list(
          order = "random", 
          components = c(point_task_names), 
          interruptions = list(
            list(
              spacing = "random",
              numInterruptions = 2,
              components = list("attention_check")
            )
          )
        ), 
        list(
          order = "random", 
          components = c(pointArc_task_names),
          interruptions = list(
            list(
              spacing = "random",
              numInterruptions = 2,
              components = list("attention_check")
            )
          )
        )
      )
    ),
    "post_study"
  )
)
# attention check 
# "interruptions": [
#         {
#           "spacing": "random",
#           "numInterruptions": 2,
#           "components": ["myAttentionCheckComponent"]
#         }
#       ]

# test 
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
  components = c(introduction_component, consent_comp, calibration_intro, training_intro, testing_intro, components, point_arc_components, post_study_component, attention_check),
  sequence=sequence
)

config_json <- final_output %>% convert_to_json(.)
# validate 
validator(config_json, verbose = TRUE)

# save output 
config_json %>% write(here("R", "config-r.json"))

