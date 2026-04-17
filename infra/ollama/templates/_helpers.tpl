{{- define "photosearch-ollama.fullname" -}}
photosearch-ollama
{{- end }}

{{- define "photosearch-ollama.labels" -}}
app.kubernetes.io/name: {{ include "photosearch-ollama.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "photosearch-ollama.selectorLabels" -}}
app.kubernetes.io/name: {{ include "photosearch-ollama.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
