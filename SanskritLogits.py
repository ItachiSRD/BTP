from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BeamScorer, BeamSearchScorer, LogitsProcessor, LogitsProcessorList
class SanskritAnushtupLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, num_beams: int, sanskrit_token_ids):
        self.tokenizer = tokenizer
        self.num_beams = num_beams
        self.sanskrit_token_ids = set(sanskrit_token_ids)
        self.generation_step = 0

    def is_anushtup_critical_position(self, syllable_index: int) -> bool:
        pada_position = syllable_index % 8
        return pada_position in [4, 5, 6]  # 5th, 6th, or 7th position in pada

    def get_required_syllable_type(self, syllable_index: int) -> bool:
        pada_position = syllable_index % 8
        pada_number = syllable_index // 8

        if pada_position == 4:  # 5th syllable
            return True  # must be laghu
        elif pada_position == 5:  # 6th syllable
            return False  # must be guru
        elif pada_position == 6:  # 7th syllable
            return pada_number % 2 == 1  # laghu for 2nd and 4th padas, guru for 1st and 3rd
        else:
            return None  # No specific requirement
        
    

    def enforce_syllable_type(self, current_text: str, should_be_laghu: bool, scores: torch.FloatTensor) -> torch.FloatTensor:
        current_syllables = get_syllables_flat_improved(current_text)
        pada_position = len(current_syllables) % 8  # Calculate the position within the pada
        pada_number = len(current_syllables) // 8   # Calculate the current pada number (1-based index)
    

        for token in range(self.tokenizer.vocab_size):
            if token not in self.sanskrit_token_ids:
                continue
            next_token = self.tokenizer.decode([token])
            candidate_text = current_text + next_token
            candidate_syllables = get_syllables_flat_improved(candidate_text)
            
            if len(candidate_syllables) == len(current_syllables) + 1:
                new_syllable = candidate_syllables[-1]
                next_syllable = ""  # We don't need to check the next syllable here

                # Check for halanta in the 6th position
                if pada_position == 5 and HALANTA in get_characters(new_syllable)[0]:
                    scores[token] = float('-inf')

                # Check for hrasva or deergha in the 7th position
                if pada_position == 6:
                    if pada_number % 2 == 1:  
                        if any(char in next_token for char in [MATRA[i] for i in [0, 2, 4, 6, 8, 9, 10, 11, 12]] + EXTENDED_MATRA + [SWARA[i] for i in [1, 3, 5, 7, 9, 10, 11, 12, 13]] + EXTENDED_SWARA):
                            scores[token] = float('-inf')
                            continue
                    else:  
                        if any(char in next_token for char in [MATRA[i] for i in [1, 3, 5, 7]] + [SWARA[i] for i in [0, 2, 4, 6, 8]]):
                            scores[token] = float('-inf')
                            continue

                if should_be_laghu == is_laghu(new_syllable, next_syllable):
                    scores[token] = 0
                else:
                    scores[token] = float('-inf')
        
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.generation_step += 1
        batch_size, seq_len = input_ids.shape
        
        for batch_idx in range(batch_size):
            current_text = self.tokenizer.decode(input_ids[batch_idx], skip_special_tokens=True)
            current_syllables = get_syllables_flat_improved(current_text)
            
            logger.info(f"\n=== Generation Step {self.generation_step}, Batch {batch_idx} ===")
            logger.info(f"Current text: {current_text}")
            logger.info(f"Current syllable count: {len(current_syllables)}")
            
            if len(current_syllables) >= 32:
                logger.info("Reached 32 syllables, allowing only EOS token")
                scores[batch_idx].fill_(float('-inf'))
                scores[batch_idx, self.tokenizer.eos_token_id] = 0
                continue
            
            is_critical_position = self.is_anushtup_critical_position(len(current_syllables))
            
            if is_critical_position:
                should_be_laghu = self.get_required_syllable_type(len(current_syllables))
                scores[batch_idx] = self.enforce_syllable_type(current_text, should_be_laghu, scores[batch_idx])
                logger.info(f"Enforced {'laghu' if should_be_laghu else 'guru'} syllable at position {len(current_syllables) + 1}")
            else:
                # For non-critical positions, we still ensure only Sanskrit tokens are used
                non_sanskrit_mask = torch.ones_like(scores[batch_idx], dtype=torch.bool)
                non_sanskrit_mask[list(self.sanskrit_token_ids)] = False
                scores[batch_idx][non_sanskrit_mask] = float('-inf')
            
            # Ensure we don't generate more than 32 syllables
            # if len(current_syllables) == 31:
            #     # For the last syllable, we need to ensure it completes the verse
            #     scores[batch_idx] = self.enforce_syllable_type(current_text, None, scores[batch_idx])
            #     logger.info("Enforcing final syllable to complete 32 syllables")
        
        return scores
