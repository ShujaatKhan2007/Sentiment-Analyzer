import { useState } from 'react'

// The backend URL is read from an environment variable so that we never
// hardcode it. Vite exposes any variable prefixed with VITE_ to the
// browser via import.meta.env.
//
// Locally: set VITE_API_URL in frontend/.env (see .env.example)
// On Vercel: set VITE_API_URL in the project's Environment Variables settings
//
// If it's not set at all, we fall back to a local backend URL so that
// `npm run dev` still works out of the box for local development.
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

function App() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null) // { sentiment, confidence }
  const [error, setError] = useState('')

  const handleAnalyze = async () => {
    const trimmedText = text.trim()

    if (!trimmedText) {
      setError('Please enter some text before analyzing.')
      setResult(null)
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: trimmedText }),
      })

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      // This covers both network errors (backend unavailable / CORS issues)
      // and non-OK HTTP responses.
      setError(
        'Could not reach the backend. Please make sure the server is running and try again.'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <div className="card">
        <h1>Sentiment Analyzer</h1>
        <p className="subtitle">
          Enter some text below and click "Analyze" to find out whether it
          sounds positive or negative.
        </p>

        <textarea
          className="textarea"
          placeholder="Type or paste your text here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={6}
        />

        <button
          className="analyze-button"
          onClick={handleAnalyze}
          disabled={loading}
        >
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>

        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <span>Analyzing sentiment...</span>
          </div>
        )}

        {error && <div className="error-message">{error}</div>}

        {result && !loading && (
          <div
            className={`result ${
              result.sentiment === 'Positive' ? 'result-positive' : 'result-negative'
            }`}
          >
            <div className="result-row">
              <span className="result-label">Sentiment:</span>
              <span className="result-value">{result.sentiment}</span>
            </div>
            <div className="result-row">
              <span className="result-label">Confidence:</span>
              <span className="result-value">
                {(result.confidence * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
