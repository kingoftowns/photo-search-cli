{{- define "photosearch-qdrant.fullname" -}}
photosearch-qdrant
{{- end }}

{{- define "photosearch-qdrant.labels" -}}
app.kubernetes.io/name: {{ include "photosearch-qdrant.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "photosearch-qdrant.selectorLabels" -}}
app.kubernetes.io/name: {{ include "photosearch-qdrant.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
