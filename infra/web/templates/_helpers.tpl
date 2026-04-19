{{- define "photosearch-web.fullname" -}}
photosearch-web
{{- end }}

{{- define "photosearch-web.labels" -}}
app.kubernetes.io/name: {{ include "photosearch-web.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: ai-photos
{{- end }}

{{- define "photosearch-web.selectorLabels" -}}
app.kubernetes.io/name: {{ include "photosearch-web.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
