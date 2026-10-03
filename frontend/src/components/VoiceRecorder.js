'use client';

import React, { useState, useRef, useEffect } from 'react';
import Box from '@mui/material/Box';
import Fab from '@mui/material/Fab';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Tooltip from '@mui/material/Tooltip';
import MicIcon from '@mui/icons-material/Mic';
import StopIcon from '@mui/icons-material/Stop';
import ReplayIcon from '@mui/icons-material/Replay';
import GraphicEqIcon from '@mui/icons-material/GraphicEq';

/**
 * Encodes audio samples to 16-bit Mono PCM WAV format (16,000 Hz)
 * ensuring direct compatibility with Wav2Vec2 and Whisper backends.
 */
function encodeWAV(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (view, offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // Mono channel
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // Byte rate
  view.setUint16(32, 2, true); // Block align
  view.setUint16(34, 16, true); // Bits per sample
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([view], { type: 'audio/wav' });
}

export default function VoiceRecorder({
  onAudioCaptured,
  isLoading = false,
  onReset,
}) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [audioUrl, setAudioUrl] = useState(null);

  const mediaStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const scriptProcessorRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);

  // Clean up timers and audio contexts on unmount
  useEffect(() => {
    return () => {
      stopTracks();
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const stopTracks = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
      scriptProcessorRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
  };

  const startRecording = async () => {
    try {
      setAudioUrl(null);
      audioChunksRef.current = [];
      setRecordingDuration(0);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      mediaStreamRef.current = stream;

      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        audioChunksRef.current.push(new Float32Array(inputData));
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      setIsRecording(true);

      // Start duration counter
      timerRef.current = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error('Microphone access failed:', err);
      alert('Could not access microphone. Please grant permission in your browser.');
    }
  };

  const stopRecording = () => {
    if (!isRecording) return;

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    setIsRecording(false);

    // Merge audio chunks into 16kHz PCM WAV
    const totalLength = audioChunksRef.current.reduce((acc, chunk) => acc + chunk.length, 0);
    const mergedSamples = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of audioChunksRef.current) {
      mergedSamples.set(chunk, offset);
      offset += chunk.length;
    }

    stopTracks();

    if (totalLength > 0) {
      const wavBlob = encodeWAV(mergedSamples, 16000);
      const previewUrl = URL.createObjectURL(wavBlob);
      setAudioUrl(previewUrl);

      if (onAudioCaptured) {
        onAudioCaptured(wavBlob, previewUrl);
      }
    }
  };

  const handleReset = () => {
    setAudioUrl(null);
    setRecordingDuration(0);
    if (onReset) onReset();
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  return (
    <Paper
      elevation={0}
      sx={{
        p: { xs: 3, md: 4 },
        textAlign: 'center',
        borderRadius: '24px',
        bgcolor: '#FFFFFF',
        border: '1px solid rgba(0, 0, 0, 0.06)',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.04)',
        maxWidth: 540,
        mx: 'auto',
      }}
    >
      {/* Prominent Floating Action Button with Material Pulse Animation */}
      <Box sx={{ position: 'relative', display: 'inline-flex', my: 2 }}>
        {/* Subtle Orange Pulsing Rings when recording */}
        {isRecording && (
          <>
            <Box
              sx={{
                position: 'absolute',
                top: -12,
                left: -12,
                right: -12,
                bottom: -12,
                borderRadius: '50%',
                bgcolor: 'primary.main',
                opacity: 0.25,
                animation: 'pulseRing 1.8s cubic-bezier(0.215, 0.61, 0.355, 1) infinite',
                '@keyframes pulseRing': {
                  '0%': { transform: 'scale(0.85)', opacity: 0.4 },
                  '50%': { transform: 'scale(1.25)', opacity: 0.15 },
                  '100%': { transform: 'scale(0.85)', opacity: 0.4 },
                },
              }}
            />
            <Box
              sx={{
                position: 'absolute',
                top: -24,
                left: -24,
                right: -24,
                bottom: -24,
                borderRadius: '50%',
                bgcolor: 'primary.light',
                opacity: 0.12,
                animation: 'pulseOuter 2.2s ease-out infinite',
                '@keyframes pulseOuter': {
                  '0%': { transform: 'scale(0.8)', opacity: 0.3 },
                  '100%': { transform: 'scale(1.4)', opacity: 0 },
                },
              }}
            />
          </>
        )}

        <Tooltip title={isRecording ? 'Click to Stop' : 'Click to Speak'} arrow>
          <span>
            <Fab
              color="primary"
              disabled={isLoading}
              onClick={isRecording ? stopRecording : startRecording}
              sx={{
                width: 80,
                height: 80,
                borderRadius: '50%',
                transition: 'all 0.3s ease',
                bgcolor: isRecording ? '#D84315' : 'primary.main',
                '&:hover': {
                  bgcolor: isRecording ? '#BF360C' : 'primary.dark',
                  transform: 'scale(1.05)',
                },
              }}
            >
              {isLoading ? (
                <CircularProgress size={36} sx={{ color: '#FFFFFF' }} />
              ) : isRecording ? (
                <StopIcon sx={{ fontSize: 38, color: '#FFFFFF' }} />
              ) : (
                <MicIcon sx={{ fontSize: 38, color: '#FFFFFF' }} />
              )}
            </Fab>
          </span>
        </Tooltip>
      </Box>

      {/* Recording State Label */}
      <Box sx={{ mt: 1.5, mb: 1 }}>
        <Typography
          variant="h6"
          sx={{
            fontWeight: 600,
            color: isRecording ? 'primary.dark' : 'text.primary',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 1,
          }}
        >
          {isRecording ? (
            <>
              <GraphicEqIcon sx={{ color: 'primary.main', animation: 'bounce 0.8s infinite alternate' }} />
              Listening... ({formatTime(recordingDuration)})
            </>
          ) : isLoading ? (
            'Analyzing Accent & Intent...'
          ) : audioUrl ? (
            'Audio Recorded'
          ) : (
            'Tap to Speak'
          )}
        </Typography>

        <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
          {isRecording
            ? 'Speak naturally about food, fashion, travel, or news.'
            : audioUrl
            ? 'Your voice query has been captured and submitted for regional inference.'
            : 'Ask in your natural accent for authentic regional recommendations.'}
        </Typography>
      </Box>

      {/* Audio Playback Preview & Re-record Controls */}
      {audioUrl && !isRecording && (
        <Box
          sx={{
            mt: 2.5,
            pt: 2,
            borderTop: '1px solid rgba(0, 0, 0, 0.06)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 1.5,
          }}
        >
          <Box component="audio" controls src={audioUrl} sx={{ width: '100%', maxWidth: 360, height: 40 }} />
          <Button
            size="small"
            variant="text"
            color="primary"
            startIcon={<ReplayIcon />}
            onClick={handleReset}
            disabled={isLoading}
            sx={{ fontWeight: 600 }}
          >
            Record Again
          </Button>
        </Box>
      )}
    </Paper>
  );
}
