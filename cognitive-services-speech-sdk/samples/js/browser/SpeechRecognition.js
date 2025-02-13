// const fs = require("fs");
// const sdk = require("microsoft-cognitiveservices-speech-sdk");
// require("dotenv").config()

// // This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"

// const speechConfig = sdk.SpeechConfig.fromSubscription(process.env.SPEECH_KEY, process.env.SPEECH_REGION);
// const audioConfig = sdk.AudioConfig.fromWavFileInput(fs.readFileSync("testing.wav"));

// var pronunciationAssessmentConfig = new sdk.PronunciationAssessmentConfig( 
//     referenceText= "", 
//     gradingSystem= sdk.PronunciationAssessmentGradingSystem.HundredMark,  
//     granularity= sdk.PronunciationAssessmentGranularity.Phoneme,  
//     enableMiscue= false); 
// // pronunciationAssessmentConfig.enableProsodyAssessment(); 
// // pronunciationAssessmentConfig.enableContentAssessmentWithTopic("greeting");  

// // speechConfig.speechRecognitionLanguage = "zh-CN";

// // function fromFile() {
// //     let speechRecognizer = new sdk.SpeechRecognizer(speechConfig, audioConfig);

// //     speechRecognizer.recognizeOnceAsync(result => {
// //         switch (result.reason) {
// //             case sdk.ResultReason.RecognizedSpeech:
// //                 console.log(`RECOGNIZED: Text=${result.text}`);
// //                 break;
// //             case sdk.ResultReason.NoMatch:
// //                 console.log("NOMATCH: Speech could not be recognized.");
// //                 break;
// //             case sdk.ResultReason.Canceled:
// //                 const cancellation = sdk.CancellationDetails.fromResult(result);
// //                 console.log(`CANCELED: Reason=${cancellation.reason}`);

// //                 if (cancellation.reason == sdk.CancellationReason.Error) {
// //                     console.log(`CANCELED: ErrorCode=${cancellation.ErrorCode}`);
// //                     console.log(`CANCELED: ErrorDetails=${cancellation.errorDetails}`);
// //                     console.log("CANCELED: Did you set the speech resource key and region values?");
// //                 }
// //                 break;
// //         }
// //         speechRecognizer.close();
// //     });
// // }
// // fromFile();


// var speechRecognizer = sdk.SpeechRecognizer.FromConfig(speechConfig, audioConfig);

// // (Optional) get the session ID
// speechRecognizer.sessionStarted = (s, e) => {
//     console.log(`SESSION ID: ${e.sessionId}`);
// };
// pronunciationAssessmentConfig.applyTo(speechRecognizer);

// speechRecognizer.recognizeOnceAsync((speechRecognitionResult) => {
//     // The pronunciation assessment result as a Speech SDK object
//     var pronunciationAssessmentResult = sdk.PronunciationAssessmentResult.fromResult(speechRecognitionResult);
//     console.log(pronunciationAssessmentResult)

//     // The pronunciation assessment result as a JSON string
//     var pronunciationAssessmentResultJson = speechRecognitionResult.properties.getProperty(sdk.PropertyId.SpeechServiceResponse_JsonResult);
//     console.log(pronunciationAssessmentResultJson)
// },
// {});

const fs = require("fs");
const sdk = require("microsoft-cognitiveservices-speech-sdk");
require("dotenv").config();

// Ensure environment variables are set
if (!process.env.SPEECH_KEY || !process.env.SPEECH_REGION) {
    console.error("Missing SPEECH_KEY or SPEECH_REGION. Please set them in your .env file.");
    process.exit(1);
}

// Initialize speech configuration
const speechConfig = sdk.SpeechConfig.fromSubscription(process.env.SPEECH_KEY, process.env.SPEECH_REGION);
speechConfig.speechRecognitionLanguage = "zh-CN"; // Adjust for your language

// Use WAV file input or microphone input
const audioConfig = sdk.AudioConfig.fromWavFileInput(fs.readFileSync("testing.wav")); 
// If you want to use a microphone instead, uncomment the line below:
// const audioConfig = sdk.AudioConfig.fromDefaultMicrophoneInput(); 

// Configure pronunciation assessment
const pronunciationAssessmentConfig = new sdk.PronunciationAssessmentConfig(
    "吃饭吃饭吃饭吃饭", // Reference text for pronunciation assessment
    sdk.PronunciationAssessmentGradingSystem.HundredMark,  
    sdk.PronunciationAssessmentGranularity.Phoneme,  
    false
);

// Initialize speech recognizer
const speechRecognizer = new sdk.SpeechRecognizer(speechConfig, audioConfig);

// Apply pronunciation assessment configuration
pronunciationAssessmentConfig.applyTo(speechRecognizer);

// Event: Recognition starts
speechRecognizer.sessionStarted = (s, e) => {
    console.log(`SESSION STARTED: ${e.sessionId}`);
};

// Event: Speech is recognized
speechRecognizer.recognized = (s, event) => {
    if (event.result.reason === sdk.ResultReason.RecognizedSpeech) {
        console.log(`Recognized Text: ${event.result.text}`);

        // Extract pronunciation assessment result
        const pronunciationAssessmentResultJson = event.result.properties.getProperty(
            sdk.PropertyId.SpeechServiceResponse_JsonResult
        );
        console.log("Pronunciation Assessment JSON:", pronunciationAssessmentResultJson);
    } else {
        console.log("Speech not recognized.");
    }
};

// Event: Recognition is canceled
speechRecognizer.canceled = (s, event) => {
    console.log(`Recognition Canceled: ${event.reason}`);
    if (event.reason === sdk.CancellationReason.Error) {
        console.log(`Error Code: ${event.errorCode}`);
        console.log(`Error Details: ${event.errorDetails}`);
    }
};

// Start speech recognition
speechRecognizer.recognizeOnceAsync(
    (result) => {
        console.log("Recognition completed.");
        speechRecognizer.close();
    },
    (err) => {
        console.error("Error starting recognition:", err);
        speechRecognizer.close();
    }
);
