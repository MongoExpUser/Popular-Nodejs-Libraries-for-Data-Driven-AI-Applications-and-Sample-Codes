/*************************************************************************************************************************************************
* classify-review-embed-translate-transcribe-free.mjs                                                                                            *                            
*                                                                                                                		                 *
* Project:  Classifications, Reviews, Embeddings and Translations  with NodeJS GenAI Free Libraries.                                             *
*                                                                             							                 *
*  Copyright © 2024. MongoExpUser.  All Rights Reserved.                                 					                 *
*                                                                                     						                 *
*  License: MIT - https://github.com/MongoExpUser/Popular-Nodejs-Libraries-for-Data-Driven-AI-Applications-and-Sample-Codes/blob/main/LICENSE    *
*                                                                                                            			                 *
*************************************************************************************************************************************************/


import { Image } from "image-js";
import { inspect } from "node:util";
import { readFileSync } from "node:fs";
import * as tf from "@tensorflow/tfjs-node";
import { pipeline } from '@xenova/transformers';
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as encoder from "@tensorflow-models/universal-sentence-encoder";


class AIApp
{
	        constructor()
	        {
	      	   return null;
	        }
	
	        async prettyPrint(value)
	        {
		   console.log(inspect(value, { showHidden: false, colors: true, depth: Infinity }));
	        }

		async separator()
		{
		   console.log(`------------------------------------------------------------------------`);
		}

		async classifyText(texts)
		{
		    console.log("");
		    console.log("-- Classify Text with Transformer.js --");

		    let classifierResults = [];
		    const textLen  = texts.length;
		    const classifier = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";
		    const cpl = await pipeline("sentiment-analysis", classifier);

		    for(let index = 0; index < textLen; index++)
		    {
		        const text = texts[index];
			const cplResult = await cpl(text);
		        classifierResults.push(cplResult);
		    }

		    return classifierResults;
		}

		async reviewText(texts)
		{
		    console.log("");
		    console.log("-- Review Text with Transformer.js --");

		    let reviewerResults = [];
		    const textLen  = texts.length;
		    const reviewer  = "Xenova/bert-base-multilingual-uncased-sentiment";
		    const rpl = await pipeline("sentiment-analysis", reviewer);

		    for(let index = 0; index < textLen; index++)
		    {
		        const text = texts[index];
			const rplResult = await rpl(text);
		        reviewerResults.push(rplResult);
		    }

		    return reviewerResults;
		}

		async generateImageEmbeddingsWithTensorFlowMobilenet(files)
		{
		    console.log("");
		    console.log("-- Generate Image Embeddings with TensorFlow.js - Mobilenet Model --");

		    let embeddings = []
		    const fileLen  = files.length;

		    for(let index = 0; index < fileLen; index++)
		    {
		        const file = files[index];
		        const image = await Image.load(file);
		        const model = await mobilenet.load(); 
		        const getEmbedding = await model.infer(image);
		        const embed = getEmbedding .arraySync()[0];
		        embeddings.push(embed);
		    }

		    return embeddings;
		}

		async generateImageEmbeddingsWithTensorFlowUse(files)
		{
		    console.log("");
		    console.log("-- Generate Image Embeddings with TensorFlow.js - Use (universal-sentence-encoder) Model --");

		    const embeddings = []
		    const fileLen  = files.length;

		    for(let index = 0; index < fileLen; index++)
		    {
		        const file = files[index];
		        const image = file; 
		        const model = await encoder.load();
		        const getEmbedding  = await model.embed(image);
		        const embed = getEmbedding .arraySync()[0];
		        embeddings.push(embed);
		    }

		    return embeddings;
		}

		async generateImageEmbeddingsWithTransformer(files)
		{
		    console.log("");
		    console.log("-- Generating Image Embeddings with Transformer.js --");

		    let embeddings = []
		    const fileLen  = files.length;
		    const imageFeatureExtractorModel = "Xenova/vit-base-patch16-224-in21k";
		    const ifepl = await pipeline("image-feature-extraction", imageFeatureExtractorModel);

		    for(let index = 0; index < fileLen; index++)
		    {
		        const file = files[index];
		        const image = file; 
		        const embed = await ifepl(image);
		        embeddings.push(embed)
		    }

		    return embeddings;
		}

		async generateTextEmbeddingsWithTransformer(texts)
		{
		    console.log("");
		    console.log("-- Generating Text Embeddings with Transformer.js --");

		    const embeddings = []
		    const textLen  = texts.length;
		    const textFeatureExtractor = "Xenova/all-MiniLM-L6-v2";
		    const tfepl = await pipeline("feature-extraction", textFeatureExtractor);

		    for(let index = 0; index < textLen; index++)
		    {
		        const text = texts[index];
		        const embed = await tfepl(text, { pooling: 'mean', normalize: true });
		        embeddings.push(embed)
		    }

		    return embeddings;
		}

		async translateText(texts, sourceLanguage, targetLanguage)
		{
		    console.log("");
		    console.log("-- Translating Text with Transformer.js --");

		    let tranlatorResult = []
		    const textLen  = texts.length;
		    const translator = "Xenova/nllb-200-distilled-600M";
		    const trpl = await pipeline('translation', translator);

		    const languageOptions = { src_lang: sourceLanguage, tgt_lang: targetLanguage };

		    for(let index = 0; index < textLen; index++)
		    {
		        const text = texts[index];
		        const trplResult = await trpl(text, languageOptions);
		        tranlatorResult.push(trplResult)
		    }

		    return tranlatorResult;
		}


	    	async readLocalOrRemoteFile(file)
	    	{
			let buffer;
			const aiapp = new AIApp();
	
			// read local or remote file
			try
			{
				buffer = readFileSync(file);  // try a local file path
			}
			catch(localFileError)
			{	
				if(localFileError)
				{
					await aiapp.prettyPrint( { "Error": `Local file not found. Will try remote file.` } );
	
					try
					{ 
						buffer = Buffer.from(await (await fetch(file)).arrayBuffer());  // try a url (remote) file path
					}
					catch(urlFileError)
					{
						if(urlFileError)
						{
							await aiapp.prettyPrint( { "Error": `Cannot read file.` } );
						}
					}
				}
			}
	
			return buffer;
		}

		async transcribeAudio(files)
		{
		    console.log("");
		    console.log("-- transcribe Audio with Transformer.js --");

		    const aiapp = new AIApp();
		    let transcribeResults = [];
		    const fileLen  = files.length;
		    const transcriber = "Xenova/whisper-tiny.en";
		    const transpl = await pipeline("automatic-speech-recognition", transcriber);
		    
		    const start = new Date();

		    for(let index = 0; index < fileLen; index++)
		    {
		        let file = files[index];
			const buffer = await aiapp.readLocalOrRemoteFile(file);
			let wf = new wavefile.WaveFile(buffer);
			wf.toBitDepth("32f");   // data type -  Float32Array
			wf.toSampleRate(16000); // sampling rate of 16000
			let data = wf.getSamples();

			if(Array.isArray(data)) 
			{
			    if(data.length > 1) 
			    {
				const scaleFactor = Math.sqrt(2);

				// merge channels to the first channel to save memory
				const data0Len = data[0].length;
				for (let i = 0; i < data0Len; ++i) 
				{
				    data[0][i] = scaleFactor * (data[0][i] + data[1][i]) / 2;
				}
			    }

			    // select the first channel and transcribe
			    let data0 = data[0];
			    let transplResult = await transpl(data0);
		            transcribeResults.push(transplResult);
			}
		    }

		    const stop = new Date();

		    const duration = (stop.getTime() - start.getTime()) / 1000;
		    console.log("===============================================================");
		    console.log("Duration of Transcription :" , duration, "seconds");
		    console.log("===============================================================");
				
		    return transcribeResults;
		}
	

		async testAIApp()
		{
		    const aiapp = new AIApp();

		    // A. define inputs
                   
		   // 1. review text
		    const textsToReview = [
		    	"The location is very far.", 
		    	"The hotel is okay."
		    ];

		    // 2. classiy text
		    const textsToClassify = [
		    	"Feeling happy today.", 
		    	"Last game was very difficult.", 
		    	"The support provided is excellent."
		    ];

		    // 3. & 4. embed image
		    const cwd = process.cwd();
		    const imageFilesToEmbed = [
		    	`${cuwd}/images/nodejs-logo.png`,
			`${cuwd}/images/python-logo.png`,
			`${cuwd}/images/rust-logo.png`
		    ];

		    // 5. embed text
		    const textsToEmbed = ["The project is going as planned."];

		    // 6. translate text
		    const textsToTranslate = [
			"What is your name.",
			"I like to walk my dog.",
			"I want to visit my mum and dad."
		    ];
		    const sourceLanguage = "eng_Latn"; // English Language
		    const targetLanguage = "yor_Latn"; // Yoruba Language
		    // link to supported languages: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

		    // 7. transcribe audio
		   // local and remote (url) files can be transcribed
		   const audioFilesToTranscibe = [
			`${cuwd}/audios/oh-yeah-everything-is-fine.wav`,
			"https://www.pacdv.com/sounds/voices/oh-yeah-everything-is-fine.wav",
			`${cuwd}/audios/jfk.wav`,
			"https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav",
			`${cuwd}/audios/Sports.wav`,
			"https://voiceage.com/wbsamples/in_stereo/Sports.wav`
		   ];


		    // B. test
		   
		    // 1. text sa-review with transformer.js
		    const textReviewTr  = await aiapp.reviewText(textsToReview);
		    await aiapp.prettyPrint( { "textReviewTransformer" : textReviewTr } );

		    // 2. text sa-classification with transformer.js
		    const textClassifyTr  = await aiapp.classifyText(textsToClassify);
		    await aiapp.prettyPrint( { "textClassifierTransformer" : textClassifyTr } );

		    // 3. image embedding with tensorflow.js
		    // a. mobilenet
		    const imageEmbeddingsTfMob = await aiapp.generateImageEmbeddingsWithTensorFlowMobilenet(imageFilesToEmbed);
		    await aiapp.prettyPrint( { "imageEmbeddingTensorFlowMob" : imageEmbeddingsTfMob } );

		    // b. use - universal sentense encoder
		    const imageEmbeddingsTfUse = await aiapp.generateImageEmbeddingsWithTensorFlowUse(imageFilesToEmbed);
		    await aiapp.prettyPrint( { "imageEmbeddingTensorFlowuse" : imageEmbeddingsTfUse } );

		    // 4. image embedding with transformer.js
		    const imageEmbeddingsTr = await aiapp.generateImageEmbeddingsWithTransformer(imageFilesToEmbed);
		    await aiapp.prettyPrint( { "imageEmbeddingTransformer" : imageEmbeddingsTr} );

		    // 5. text embedding with transformer.js
		    const textEmbeddingsTr  = await aiapp.generateTextEmbeddingsWithTransformer(textsToEmbed);
		    await aiapp.prettyPrint( { "textEmbeddingTransformer" : textEmbeddingsTr } );

		    // 6. text translation with transformer.js
		    const textTranslateTr = await aiapp.translateText(textsToTranslate, sourceLanguage, targetLanguage);
		    await aiapp.prettyPrint( { "textTranslatorTransformer" : textTranslateTr } );
		    /*
			// results - perfect translation - excellent model
			textTranslatorTransformer: [
				[ { translation_text: 'Kí ni orúkọ rẹ?' } ],
				[ { translation_text: 'Mo máa ń fẹ́ láti rìnrìn àjò fún ajá mi.' } ],
				[ { translation_text: 'Mo fẹ́ lọ kí àwọn òbí mi.' } ]
	  		]
	  	   */

		   // 7. transcribe with transformer.js
		   const audioTranscibeTr = await aiapp.transcribeAudio(audioFilesToTranscibe);
		   await aiapp.prettyPrint( { "audioTranscibeTr" : audioTranscibeTr } );
		   /*
     			// results of transcription - excellent model
	     		===============================================================
			Duration of Transcription : 16.436 seconds
			===============================================================
			{
			  audioTranscibeTr: [
			    { text: " Oh yeah, everything's fine." },
			    { text: " Oh yeah, everything's fine." },
			    { text: ' And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.' },
			    { text: ' And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.' },
			    { text: " Good evening ladies and gentlemen. We'd like to welcome you to new Daniel Radios broadcast of the" },
			    { text: " Good evening ladies and gentlemen. We'd like to welcome you to new Daniel Radios broadcast of the" }
			  ]
			}
     		  */
		}
}


const aiapp = new AIApp();
await aiapp.testAIApp();
