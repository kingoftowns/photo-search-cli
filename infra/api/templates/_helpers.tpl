{{- define "photosearch-api.fullname" -}}
photosearch-api
{{- end }}

{{- define "photosearch-api.labels" -}}
app.kubernetes.io/name: {{ include "photosearch-api.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: ai-photos
{{- end }}

{{- define "photosearch-api.selectorLabels" -}}
app.kubernetes.io/name: {{ include "photosearch-api.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
