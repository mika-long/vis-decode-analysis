library(jsonlite)
library(purrr)
library(jsonvalidate)
library(httr)
library(here)
library(tidyverse)

# --- initialize json validator
# --- only works when one have access to the internet
schema_url <- "https://raw.githubusercontent.com/revisit-studies/study/v2.4.1/src/parser/StudyConfigSchema.json"
schema <- content(GET(schema_url), as = "text")
validator <- json_validator(schema, engine = "ajv")

# --- Get the list of numbers to pull from
exp_json_data <- jsonlite::fromJSON(here(
  "data",
  "moritz",
  "stimuli",
  "exp2-stimuli.json"
)) %>%
  as_tibble(.)

stimuli_df <- exp_json_data %>%
  select(index, seed, noise, flip, type, data) %>%
  mutate(noise = as.character(noise))

point_ids <- stimuli_df %>% filter(type == "point") %>% pull(index)
pointArc_ids <- stimuli_df %>% filter(type == "point_arc") %>% pull(index)

block5_ids <- 1:10

# First 2 of each type used as cross-type training:
# point participants train on pointArc charts, and vice versa
point_train_indices <- point_ids[1:2]
pointArc_train_indices <- pointArc_ids[1:2]

# Exclude training indices from test pools
point_ids_test <- setdiff(point_ids, point_train_indices)
pointArc_ids_test <- setdiff(pointArc_ids, pointArc_train_indices)

get_true_mean <- function(idx) {
  stimuli_df %>%
    filter(index == idx) %>%
    pull(data) %>%
    .[[1]] %>%
    pull(y) %>%
    mean() %>%
    round(2)
}

#--- Define UI and metadata

metadata <- list(
  title = "Visual Decoding Operators - Average Estimation in Line Graphs",
  version = "pilot",
  authors = list("Sheng Long"),
  date = "2025-12-10",
  description = "A study designed to elicit responses from participants",
  organizations = list("Northwestern University")
)

ui_config = list(
  contactEmail = "shenglong@u.northwestern.edu",
  helpTextPath = "vis-decode-retrieve-value/assets/help.md",
  logoPath = "revisitAssets/revisitLogoSquare.svg",
  withProgressBar = TRUE,
  autoDownloadStudy = FALSE,
  withSidebar = FALSE,
  urlParticipantIdParam = "PROLIFIC_PID",
  studyEndMsg = "**Thank you for completing the study.** \n\n You may click this link and return to Prolific: [https://app.prolific.com/submissions/complete?cc=C1O3ZUHV](https://app.prolific.com/submissions/complete?cc=C1O3ZUHV)"
)


#--- define functions
StudyResponse <- function(
  id,
  type,
  prompt,
  required = FALSE,
  hidden = FALSE,
  ...
) {
  list(
    id = id,
    prompt = prompt,
    required = required,
    type = type,
    hidden = hidden,
    ...
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
  response_list = NULL
) {
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

Component <- function(baseComponent, parameters, ...) {
  list(
    baseComponent = baseComponent,
    parameters = parameters,
    ...
  )
}

convert_to_json <- function(c) {
  toJSON(c, auto_unbox = TRUE, pretty = TRUE)
}

# --- construct responses for Moritz
# numericResponse uses location = "sidebar" so that the FeedbackAlert renders
# in belowStimulus during training without showing a visible input box
moritz_num_response = StudyResponse(
  id = "numericResponse",
  prompt = "Numeric response of slider to Mortiz question",
  required = FALSE,
  type = "numerical",
  hidden = FALSE,
  location = "sidebar"
)
moritz_px_response = StudyResponse(
  id = "pixelResponse",
  prompt = "Numeric response of slider to Mortiz question",
  required = FALSE,
  type = "numerical",
  hidden = TRUE
)

# --- construct basecomponent for Moritz
moritz_component <- BaseComponent(
  type = "react-component",
  path = "vis-decode-retrieve-value/assets/Moritz.tsx",
  instruction = "***Experiment Instructions.*** Please read the following paragraphs carefully. You will be asked questions about the information in the paragraphs. \n\n***Scenario:*** Assume that you are a stock market investor. You are investing your own money in stocks, and you want to determine the average price of a stock over time in order to pick the best investment.\n\n***Task:*** In this experiment, you will be shown graphs of stock prices over a one-year period like the one below. Your task is to determine the average stock price for that year. **What is the average stock price?** \n\n***Response:*** To indicate the average stock price, use your mouse to click and drag the handle on the slider, which will show a dotted red line. Move the line to where you think the average stock price is for that year. You can readjust the line by clicking, dragging, or using the left and right arrow keys. Once you are happy with your judgment of the average stock price, click the next button.",
  instructionLocation = "aboveStimulus",
  nextButtonLocation = "belowStimulus",
  parameters = list(taskIndex = "", taskType = ""),
  response_list = list(moritz_num_response, moritz_px_response)
)

# --- construct responses for Block5
# slider-x/y use location = "sidebar" so FeedbackAlert renders without visible inputs
block5_x_numresponse = StudyResponse(
  id = "slider-x",
  prompt = "Numeric response of slider to Block5",
  required = FALSE,
  type = "numerical",
  hidden = FALSE,
  location = "sidebar"
)

block5_y_numresponse = StudyResponse(
  id = "slider-y",
  prompt = "Numeric response of slider to Block5",
  required = FALSE,
  type = "numerical",
  hidden = FALSE,
  location = "sidebar"
)

block5_x_pxresponse = StudyResponse(
  id = "location-x",
  prompt = "Numeric response of slider to Block5",
  required = FALSE,
  type = "numerical",
  hidden = TRUE
)

block5_y_pxresponse = StudyResponse(
  id = "location-y",
  prompt = "Numeric response of slider to Block5",
  required = FALSE,
  type = "numerical",
  hidden = TRUE
)

block5_component <- BaseComponent(
  type = "react-component",
  path = "vis-decode-retrieve-value/assets/Block5.tsx",
  instruction = "Click on the sliders to initialize your selection. \n\n Drag the sliders such that the two balls on the x-axis and y-axis match the red dot.",
  instructionLocation = "aboveStimulus",
  nextButtonLocation = "belowStimulus",
  parameters = list(taskType = "Block5", training = ""),
  response_list = list(
    block5_x_numresponse,
    block5_x_pxresponse,
    block5_y_numresponse,
    block5_y_pxresponse
  )
)

# --- video for block 5
train_video_Block5 <- list(
  type = "markdown",
  path = "vis-decode-retrieve-value/assets/training-task5.md",
  response = list()
)

# --- Block5 training components
block5_train_params <- tibble(
  x = c(0.2, -1.5),
  y = c(0.75, 0.4)
)

block5_train_components <- block5_train_params %>%
  pmap(function(x, y) {
    Component(
      baseComponent = "Block5",
      parameters = list(
        taskType = "Block5",
        training = TRUE,
        x = x,
        y = y
      ),
      correctAnswer = list(
        list(id = "slider-x", answer = x),
        list(id = "slider-y", answer = y)
      ),
      provideFeedback = TRUE,
      allowFailedTraining = TRUE,
      trainingAttempts = 1
    )
  }) %>%
  set_names(paste0("block5_train_", seq_len(nrow(block5_train_params))))

block5_test_components <- block5_ids %>%
  imap(
    ~ Component(
      baseComponent = "Block5",
      parameters = list(taskType = "Block5", training = FALSE)
    )
  ) %>%
  set_names(paste0("block5_test_", seq_along(block5_ids)))

# --- Moritz training components
# point participants train on pointArc charts
moritz_train_point_components <- tibble(
  taskIndex = pointArc_train_indices,
  taskType = "pointArc",
  trueMean = map_dbl(pointArc_train_indices, get_true_mean)
) %>%
  pmap(function(taskIndex, taskType, trueMean) {
    Component(
      baseComponent = "Moritz",
      parameters = list(
        taskIndex = taskIndex,
        taskType = taskType,
        training = TRUE
      ),
      correctAnswer = list(list(id = "numericResponse", answer = trueMean)),
      provideFeedback = TRUE,
      allowFailedTraining = TRUE,
      trainingAttempts = 1
    )
  }) %>%
  set_names(paste0("moritz_train_point_", 1:2))

# pointArc participants train on point charts
moritz_train_pointArc_components <- tibble(
  taskIndex = point_train_indices,
  taskType = "point",
  trueMean = map_dbl(point_train_indices, get_true_mean)
) %>%
  pmap(function(taskIndex, taskType, trueMean) {
    Component(
      baseComponent = "Moritz",
      parameters = list(
        taskIndex = taskIndex,
        taskType = taskType,
        training = TRUE
      ),
      correctAnswer = list(list(id = "numericResponse", answer = trueMean)),
      provideFeedback = TRUE,
      allowFailedTraining = TRUE,
      trainingAttempts = 1
    )
  }) %>%
  set_names(paste0("moritz_train_pointArc_", 1:2))

# --- Moritz test components (training indices excluded)
point_task_names <- paste0("point_task_", seq_along(point_ids_test))
pointArc_task_names <- paste0("pointArc_task_", seq_along(pointArc_ids_test))

components <- point_ids_test %>%
  imap(
    ~ Component(
      baseComponent = "Moritz",
      parameters = list(taskIndex = .x, taskType = "point")
    )
  ) %>%
  set_names(point_task_names)

point_arc_components <- pointArc_ids_test %>%
  imap(
    ~ Component(
      baseComponent = "Moritz",
      parameters = list(taskIndex = .x, taskType = "pointArc")
    )
  ) %>%
  set_names(pointArc_task_names)

# --- Static components

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

consent_comp = list(
  consent = list(
    type = "markdown",
    path = "vis-decode-retrieve-value/assets/consent.md",
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
    nextButtonEnableTime = 3000,
    style = list(margin = "0 auto", width = "50%")
  )
)

testing_intro = list(
  testing_intro = list(
    type = "markdown",
    path = "vis-decode-retrieve-value/assets/testing_intro.md",
    response = list(),
    nextButtonEnableTime = 3000,
    style = list(margin = "0 auto", width = "50%")
  )
)

post_study_component = list(
  post_study = list(
    type = "questionnaire",
    response = list(
      list(
        id = "gender",
        prompt = "## Thank you! \n\n Please answer the following demographics related questions: \n\n Gender",
        location = "belowStimulus",
        type = "radio",
        options = c("Male", "Female", "Prefer not to say"),
        withOther = TRUE
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

# --- Sequence
# Training for Moritz is nested inside each latinSquare arm so participants
# only see cross-type training charts matching their assigned condition

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
    "training_intro",
    # Block5 training and test
    list(
      order = "fixed",
      components = list(
        "train-video-5",
        "block5_train_1",
        "block5_train_2",
        list(
          components = c(paste0("block5_test_", 1:10)),
          order = "random"
        )
      )
    ),
    "testing_intro",
    # Moritz training (cross-type) and test, assigned via latinSquare
    list(
      order = "latinSquare",
      numSamples = 1,
      components = list(
        list(
          order = "fixed",
          components = list(
            "moritz_train_point_1",
            "moritz_train_point_2",
            list(
              order = "random",
              components = c(point_task_names),
              interruptions = list(list(
                spacing = "random",
                numInterruptions = 2,
                components = list("attention_check")
              ))
            )
          )
        ),
        list(
          order = "fixed",
          components = list(
            "moritz_train_pointArc_1",
            "moritz_train_pointArc_2",
            list(
              order = "random",
              components = c(pointArc_task_names),
              interruptions = list(list(
                spacing = "random",
                numInterruptions = 2,
                components = list("attention_check")
              ))
            )
          )
        )
      )
    ),
    "post_study"
  )
)

# --- Study rules

studyRules = list(
  display = list(
    minHeight = 400,
    minWidth = 935
  ),
  browsers = list(
    allowed = list(
      list(name = "chrome", minVersion = 100),
      list(name = "firefox", minVersion = 100),
      list(name = "safari", minVersion = 10)
    ),
    blockedMessage = "You must be on a relatively modern browser, Chrome > 100, Firefox > 100, Safari > 10."
  ),
  devices = list(
    allowed = list("desktop"),
    blockedMessage = "This study requires a desktop device."
  )
)

# --- Render

final_output = list(
  `$schema` = "https://raw.githubusercontent.com/revisit-studies/study/v2.4.0/src/parser/StudyConfigSchema.json",
  studyMetadata = metadata,
  uiConfig = ui_config,
  studyRules = studyRules,
  importedLibraries = list("virtual-chinrest"),
  baseComponents = list(
    Moritz = moritz_component,
    Block5 = block5_component
  ),
  components = c(
    introduction_component,
    consent_comp,
    calibration_intro,
    training_intro,
    testing_intro,
    list(`train-video-5` = train_video_Block5),
    block5_train_components,
    block5_test_components,
    moritz_train_point_components,
    moritz_train_pointArc_components,
    components,
    point_arc_components,
    post_study_component,
    attention_check
  ),
  sequence = sequence
)

config_json <- final_output %>% convert_to_json(.)

# validate
validator(config_json, verbose = TRUE)

# save output
config_json %>% write(here("R", "config-r.json"))
