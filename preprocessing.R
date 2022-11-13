library(dplyr)
library(tidyverse)
library(lubridate)

df <- test_chess_data


df <-(df %>%
        #Pivot the data along "moves" 
  pivot_longer(cols = starts_with(c("Move_Ply", "Eval_ply", "Clock_ply")), 
               names_to = c(".value", "move"),
               names_pattern = "(.+)_(.+)", values_transform = as.factor)
  #Remove null values 
  #%>%  filter(!is.na(Eval_ply), !is.na(Move_ply), Eval_ply != "", Move_ply != "") 
  %>% mutate(move = as.numeric(move))
  %>% filter(move <= 100)
  
  #Keep track of what is moved and additional features like whether stockfish detects checkmate
  %>%   mutate(king_moved = ifelse(str_detect(Move_ply, "K"), 1, 0))
  %>%   mutate(knight_moved = ifelse(str_detect(Move_ply, "N"), 1, 0))
  %>%   mutate(queen_moved = ifelse(str_detect(Move_ply, "Q"), 1, 0))
  %>%   mutate(bishop_moved = ifelse(str_detect(Move_ply, "B"), 1, 0))
  %>%   mutate(rook_moved = ifelse(str_detect(Move_ply, "R"), 1, 0))
  %>%   mutate(is_check = ifelse(str_detect(Move_ply, "\\+"), 1, 0))
  %>%   mutate(is_checkmate = ifelse(str_detect(Move_ply, "#"), 1, 0))
  %>%   mutate(is_castle = ifelse(str_detect(Move_ply, "O"), 1, 0))
  %>%   mutate(is_capture = ifelse(str_detect(Move_ply, "x"), 1, 0))
  %>%   mutate(pawn_promotion = ifelse(str_detect(Move_ply, "="), 1, 0))
  %>%   mutate(black_has_mate = ifelse(str_detect(Eval_ply, "#-"), 1, 0))
  %>%   mutate(white_has_mate = ifelse(str_detect(Eval_ply, "#") & black_has_mate == 0, 1, 0))
  %>%   mutate(pawn_moved = ifelse(length(Move_ply) > 0 & king_moved == 0 & knight_moved == 0 & queen_moved ==0
                                   & bishop_moved == 0 & rook_moved == 0, 1, 0))

  
)

#Handles destination square of move 
df <- (df 
  %>%   mutate(destination = case_when(is_castle == 1 ~ substr(Move_ply, 0,nchar(as.character(Move_ply))), 
                                       pawn_promotion == 1 ~ substr(Move_ply, 0, 1), 
                                       is_check == 1 | is_checkmate == 1  ~ substr(Move_ply, nchar(as.character(Move_ply))-2, 
                                                                                   nchar(as.character(Move_ply))-1), 
                                       TRUE  ~ substr(Move_ply, nchar(as.character(Move_ply))-1, 
                                                      nchar(as.character(Move_ply))), 
                    )
    )
  
  #Normalize evaluation to integer values, avoiding the existence of different format when checkmate is detected
  %>% mutate(eval_normalized = case_when(black_has_mate == 1 ~ -100, 
                                         white_has_mate == 1 ~ 100, 
                                         TRUE ~ as.numeric(as.character(Eval_ply)))
              )
  %>% mutate(black_mate_in = ifelse(black_has_mate == 1, substr(Eval_ply, 3, nchar(as.character(Eval_ply))), "100")
                              )
  %>%    mutate(white_mate_in = ifelse(white_has_mate == 1, substr(Eval_ply, 2, nchar(as.character(Eval_ply))), "100")
              )   
  %>% mutate(seconds_remaining = as.numeric(hms(Clock_ply)))
  %>% mutate(white_wins = case_when(Result == "0-1" ~ 0,
                                  Result == "1/2-1/2" ~ 0.5, 
                                    Result == "1-0"~ 1))
) 

df <- (df %>%
         group_by(Index) %>%
         mutate(lag_eval = lag(eval_normalized, n=1, order_by=move))
       %>% ungroup() 
       )

df <- (df %>%
  mutate(move = as.numeric(move))
 %>% arrange(Index, move)
 %>%select("Index", "move", everything())
)
 
#Replace NA and NaN with zero, remove irrelevant columns 
df  <- subset(df, select= -c(Black, BlackRatingDiff, Black, Date, Opening, Result, Round, Site, UTCDate, UTCTime, White, 
                      WhiteRatingDiff, Move_ply, Eval_ply, Clock_ply))



write.csv(df,"for_pandas.csv")




