library(jsonlite)

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
  withSiderbar = FALSE,
  minHeightSize = 400,
  minWIdthSIze = 400, 
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

my_responses <- list(
  StudyResponse(
    id = "pixelsPerMM", 
    prompt = "Calibration results", 
    type = "reactive", # Note: 'reactive' is a specific string here, not an R concept
    required = TRUE, 
    hidden = TRUE
  )
)

# C. Create the Component
card_component <- BaseComponent(
  type = "react-component",
  path = "vis-decode-click/assets/VirtualChinrestCalibration.tsx",
  nextButtonLocation = "belowStimulus",
  parameters = list(taskid = "pixelsPerMM"),
  response_list = my_responses
)

# 4. RENDER
# We convert to JSON only at the very end.
# auto_unbox = TRUE handles the single strings.
json_output <- toJSON(card_component, auto_unbox = TRUE, pretty = TRUE)


final_output = list(
  studyMetadata = metadata, 
  uiConfig = ui_config,
  importedLibraries = list()
)

toJSON(final_output, auto_unbox = TRUE, pretty = TRUE)

