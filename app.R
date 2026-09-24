# Loading Web App related libraries
library(shiny)
library(htmltools)

# Loading Python interaction related libraries
library(reticulate)

# Loading others
library(dplyr)

# Creating UI function
ui <- htmlTemplate(
  filename = 'www/index.html'
)

# Creating Server function
server <- function(input, output, session){

}

shinyApp(ui, server)
