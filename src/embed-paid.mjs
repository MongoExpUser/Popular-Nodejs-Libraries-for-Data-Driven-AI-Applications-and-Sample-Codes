
/************************************************************************************************************************************************
*  embed-bedrock-paid.mjs                                                                                                                        *
*************************************************************************************************************************************************
*                                                                                                                                               *
*  Project: Create Text and Image Embeddings with a NodeJS GenAI Paid Library (AWS Bedrock)                                                     *
*                                                                                                                                               *
*  Copyright © 2024. MongoExpUser.  All Rights Reserved.                                                                                        *
*                                                                                                                                               *
*  License: MIT - https://github.com/MongoExpUser/Popular-Nodejs-Libraries-for-Data-Driven-AI-Applications-and-Sample-Codes/blob/main/LICENSE   *
*                                                                                                                                               *
*                                                                                                                                               *
*   This script implements a class for generating text and image embeddings                                                                     *
*                                                                                                                                               *
*                                                                                                                                               *
************************************************************************************************************************************************/




import { inspect } from "node:util";
import { readFileSync } from "node:fs";
import { BedrockClient, ListFoundationModelsCommand } from "@aws-sdk/client-bedrock";
import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";


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

  	async listBedrockFoundationModels(credentials)
  	{
  		const aiapp = new AIApp();
  		const client = new BedrockClient({credentials});
  		const command = new ListFoundationModelsCommand({});
  		const response = await client.send(command);
  		const models = response.modelSummaries;
  		await aiapp.prettyPrint( { "List of AWS Bedrock Foundation Models" : models } );
  
  		const activeModels = (models.filter((mod) => mod.modelLifecycle.status === "ACTIVE")).length;
  		const legacyModels = (models.filter((mod) => mod.modelLifecycle.status === "LEGACY")).length;
  
  		await aiapp .separator()
  		console.log(`There are ${activeModels} active and ${legacyModels} legacy foundation models in ${credentials.region}.`);
  		await aiapp .separator()
  		return response;
  	};

	async generateImageEmbeddingsWithBedrock(imageFilesToEmbed, credentials, outputEmbeddingLength)
	{
		// Supported use cases – Image search for all kinds of image identification 
	    	console.log("");
    		console.log("-- Generating Image Embeddings with AWS Bedrock's titan-embed-image-v1  --");
    		const modelId = "amazon.titan-embed-image-v1";
		const modelName = "Titan Multimodal Embeddings G1";  
		const contentType = "application/json";
    		const client = new BedrockRuntimeClient({credentials});
    		let embeddings = [];
    		const files = imageFilesToEmbed;
    		const fileLen  = files.length;

  		try 
  		{
  		    for(let index = 0; index < fileLen; index++)
  		    {
  		        let file = files[index];
  		        const imageData = readFileSync(file);
  				const base64Image = imageData.toString("base64");
  				const inputs = {
  					modelId: modelId,
  				  	contentType: contentType,
  			    		body:  JSON.stringify({ "inputImage": base64Image, "embeddingConfig": { "outputEmbeddingLength": outputEmbeddingLength } })
  				};
  		        const command = new InvokeModelCommand(inputs);
  			const response = await client.send(command);
  			const responseBody = JSON.parse(new TextDecoder().decode(response.body));
  			const embedding = responseBody.embedding;
  			const inputTextTokenCount = responseBody.inputTextTokenCount;
  			const output = { embeddingSize: embedding.length,  imageEmbedding: embedding, inputTextTokenCount: inputTextTokenCount, modelName: modelName, modelId: modelId };
  		        embeddings.push(output);
  		    } 
  		}
  		catch (error) 
  		{
  			return console.log(`Could not invoke ${modelId}. Error: ${error}`);
  		}

	    return embeddings;
	}

	async generateTextEmbeddingsWithBedrock(textsToEmbed, credentials, outputEmbeddingLength)
	{
		// Supported use cases – Text search, recommendation, and personalization.
		console.log("");
    		console.log("-- Generating Text Embeddings with AWS Bedrock's titan-embed-image-v1  --");
    		const modelId = "amazon.titan-embed-image-v1";
		const modelName = "Titan Multimodal Embeddings G1";  
		const contentType = "application/json";
    		const client = new BedrockRuntimeClient({credentials});
    		let embeddings = [];
    		const texts = textsToEmbed;
    		const textsLen  = texts.length;
    		console.log({textsLen:textsLen})

  		try 
  		{
  		    for(let index = 0; index < textsLen; index++)
  		    {
  		        let text = texts[index];
  		        const inputs = {
  				modelId: modelId,
  				contentType: contentType,
  			    	body:  JSON.stringify({ "inputText": text, "embeddingConfig": { "outputEmbeddingLength": outputEmbeddingLength } })
  			};
  		        const command = new InvokeModelCommand(inputs);
  			const response = await client.send(command);
  			const responseBody = JSON.parse(new TextDecoder().decode(response.body));
  			const embedding = responseBody.embedding;
  			const inputTextTokenCount = responseBody.inputTextTokenCount;
  			const output = { embeddingSize: embedding.length,  textEmbedding: embedding, inputTextTokenCount: inputTextTokenCount, modelName: modelName, modelId: modelId };
  		        embeddings.push(output);
  		    }
  
  		    return embeddings;
  		} 
  		catch (error) 
  		{
  			return console.log(`Could not invoke ${modelId}. Error: ${error}`);
  		}
	}


	async generateTextEmbeddingsWithBedrockTitanTextv2(textsToEmbed, credentials, outputEmbeddingLength)
	{
		// Supported use cases – Text search, recommendation, and personalization.
		console.log("");
    		console.log("-- Generating Text Embeddings with AWS Bedrock's titan-embed-text-v2:0  --"); 
    		const modelId = "amazon.titan-embed-text-v2:0"; 
		const modelName = "Titan Text Embeddings V2";  
		const contentType = "application/json";
    		const client = new BedrockRuntimeClient({credentials});
    		let embeddings = [];
    		const texts = textsToEmbed;
    		const textsLen  = texts.length;
    		console.log({textsLen:textsLen})

		try 
		{
		    for(let index = 0; index < textsLen; index++)
		    {
		        let text = texts[index];
		        const inputs = {
				modelId: modelId,
				contentType: contentType,
			    	body:  JSON.stringify({ "inputText": text, "dimensions": outputEmbeddingLength })
			};
		        const command = new InvokeModelCommand(inputs);
			const response = await client.send(command);
			const responseBody = JSON.parse(new TextDecoder().decode(response.body));
			const embedding = responseBody.embedding;
			const inputTextTokenCount = responseBody.inputTextTokenCount;
			const output = { embeddingSize: embedding.length, textEmbedding: embedding, inputTextTokenCount: inputTextTokenCount, modelName: modelName, modelId: modelId };
		        embeddings.push(output);
		    }

		    return embeddings;
		} 
		catch (error) 
		{
			return console.log(`Could not invoke ${modelId}. Error: ${error}`);
		}
	}

	
	async testAIApp()
	{
	    const aiapp = new AIApp();
	    const cuwd =  process.cwd();
		
	    const listAvailableFoundationalModel = false;
	    const generateImageEmbeddings = true;
	    const generateTextEmbeddings = false;
	    const generateTextEmbeddingsTitanTextv2 = false;
		
	    const credentialJsonFilePath = `${cuwd}/credentials.json`;
      	    const credentials =  JSON.parse(readFileSync(credentialJsonFilePath)).awsCredentials;

	    // A. define inputs

	    // 1. embed images
	    const imageFilesToEmbed = [
		`${cuwd}/images/nodejs-logo.png`,
		`${cuwd}/images/python-logo.png`,
		`${cuwd}/images/rust-logo.png`
	    ];
	    const imageOutputEmbeddingLength = 256; // 256 || 384 || 1024
	    // default ouput embedding length  = 1024;
	    // can customize to a lower value e.g 256 or 384 but smaller values are less detailed but can improve the response time

	    // 2a. & 2b. embed texts
	    const textsToEmbed1 = ["The project is going as planned."];
	    const textsToEmbed = [
  		"What is your name.",
  		"I like to walk my dog.",
  		"I want to visit my mum and dad.",
  		"What are the different services that you offer.",
  		"Dr. Roger Butler (1927-2005) is the father of SAGD."
	    ];
	    const textOutputEmbeddingLength = 256;
	    // default ouput embedding length  = 1024;
	    // can customize to a lower  value e.g 256 or 384 but smaller values are less detailed but can improve the response time

	  
	    // B. test
	    // 1. image embedding with aws bedrock's amazon.titan-embed-image-v1 model
	    if(generateImageEmbeddings  === true)
	    {
	    	const imageEmbeddingsBedrock = await aiapp.generateImageEmbeddingsWithBedrock(imageFilesToEmbed, credentials, imageOutputEmbeddingLength);
	    	await aiapp.prettyPrint( { "imageEmbeddingsBedrock" : imageEmbeddingsBedrock} );
	    }

	    // 2a. text embedding with aws bedrock's amazon.titan-embed-image-v1 model
	    if(generateTextEmbeddings === true)
	    {
	    	const textEmbeddingsBedrock = await aiapp.generateTextEmbeddingsWithBedrock(textsToEmbed, credentials, textOutputEmbeddingLength);
	    	await aiapp.prettyPrint( { "textEmbeddingsBedrock" : textEmbeddingsBedrock } );
	    }

	    // 2b. text embedding with aws bedrock's titan-embed-text-v2:0 model
	    if(generateTextEmbeddingsTitanTextv2 === true)
	    {
	    	const textEmbeddingsBedrock = await aiapp.generateTextEmbeddingsWithBedrockTitanTextv2(textsToEmbed, credentials, textOutputEmbeddingLength);
	    	await aiapp.prettyPrint( { "textEmbeddingsBedrock" : textEmbeddingsBedrock } );
	    }

	    // 3. list available aws bedrock's fundation nmodels
	    if(listAvailableFoundationalModel === true)
	    {
	    	const allModels = await aiapp.listBedrockFoundationModels(credentials);
	    }
	}
}


const aiapp = new AIApp();
await aiapp.testAIApp();

