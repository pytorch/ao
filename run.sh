###### int4wo + dyn ######

# echo "int4wo dyn gptq"
# wikitext: {'word_perplexity,none': 12.812570926788453, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.6111408960535434, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.6880826648386306, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}

# echo "int4wo dyn"
# wikitext: {'word_perplexity,none': 13.231905415833804, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.6208730195318632, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.6967710734172881, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}


###### int4wo ######

# echo "int4wo gptq"
# python /home/cdhernandez/local/ao/test/quantization/test_quant_api.py -k test_gptq_quantizer_int4wo
# echo "int4wo gptq"
# # wikitext: {'word_perplexity,none': 12.798085443034763, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.6108001089599417, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.687777474985057, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}


# echo "int4wo"
# python /home/cdhernandez/local/ao/test/quantization/test_quant_api.py -k test_quantizer_int4wo
# echo "int4wo"
# # wikitext: {'word_perplexity,none': 13.158980323604943, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.619198724726618, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.6952800588854063, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}

###### 8da4w ######

# echo "8da4w gptq"
# python /home/cdhernandez/local/ao/test/quantization/test_quant_api.py -k test_8da4w_gptq_quantizer
# echo "8da4w gptq"
# # wikitext: {'word_perplexity,none': 15.652681759772047, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.6726076112218886, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.7420990330980423, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}

# echo "8da4w"
# python /home/cdhernandez/local/ao/test/quantization/test_quant_api.py -k test_8da4w_quantizer_eval
# echo "8da4w"
# # wikitext: {'word_perplexity,none': 13.331669008220858, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.6231513927415857, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.6987975675823629, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}

###### 8da4w no dynamic act ######

# echo "8da4w gptq"
# # wikitext: {'word_perplexity,none': 12.826275573373177, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.6114630248462083, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.6883710860189309, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}

# echo "8da4w"
# wikitext: {'word_perplexity,none': 13.284947841658525, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.6220861196463254, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.6978504170203109, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}
###### 8da4w gptq ignore act quant ######

###### 8da4w experiments ######

# echo "8da4w gptq ignore act"
# wikitext: {'word_perplexity,none': 16.847579811963474, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.6957766378049428, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.7619461553063979, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}

###### new activation ######

# echo "8da4w new dynamic_act"
# python /home/cdhernandez/local/ao/test/quantization/test_quant_api.py -k test_8da4w_quantizer_eval
# echo "8da4w new dynamic_act"
# # wikitext: {'word_perplexity,none': 13.404829017406016, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.6248134072268074, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.7002740492640743, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}


# echo "8da4w new dynamic_act gptq"
# python /home/cdhernandez/local/ao/test/quantization/test_quant_api.py -k test_8da4w_gptq_quantizer
# echo "8da4w new dynamic_act gptq"
# # wikitext: {'word_perplexity,none': 186.35249273879637, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 2.658055424531603, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 1.4103711873316307, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}

# # echo "int4wo newdyn gptq"
# # python /home/cdhernandez/local/ao/test/quantization/test_quant_api.py -k test_gptq_quantizer_int4wo
# # echo "int4wo newdyn gptq"
# wikitext: {'word_perplexity,none': 12.861107615954907, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.6122804971230007, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.6891027591294213, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}


# echo "int4wo newdyn"
# python /home/cdhernandez/local/ao/test/quantization/test_quant_api.py -k test_quantizer_int4wo
# echo "int4wo newdyn"
# # wikitext: {'word_perplexity,none': 13.386525729748959, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.6243982948142355, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.6999054179299502, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}
